import abc

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import Helpers

from utils import iterator_utils
from utils import misc_utils as utils


class BaseModel(object):
    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 source_vocab, target_vocab,
                 reversed_target_vocab=None,
                 scope=None,
                 single_cell_fn=None):

        self.iterator = iterator
        self.mode = mode
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.hparams = hparams
        self.time_major = hparams.time_major

        initializer = Helpers.get_initializer(self.hparams.init_op, self.hparams.random_seed, self.hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        self.init_embeddings(scope)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(self.hparams.tgt_vocab_size, use_bias=False,
                                                      name="output_projection")

        self.single_cell_fn = single_cell_fn

        graph_res = self.build_graph(scope=scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = graph_res[1]
            self.word_count = tf.reduce_sum(self.iterator.source_sequence_length) \
                              + tf.reduce_sum(self.iterator.target_sequence_length)
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = graph_res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits, _, self.final_context_state, self.sample_id = graph_res
            self.sample_words = reversed_target_vocab.lookup(tf.to_int64(self.sample_id))

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.predict_count = tf.reduce_sum(
                self.iterator.target_sequence_length)

        warmup_steps = hparams.learning_rate_warmup_steps
        warmup_factor = hparams.learning_rate_warmup_factor
        print("  start_decay_step=%d, learning_rate=%g, decay_steps %d, "
              "decay_factor %g, learning_rate_warmup_steps=%d, "
              "learning_rate_warmup_factor=%g, starting_learning_rate=%g" %
              (hparams.start_decay_step, hparams.learning_rate, hparams.decay_steps,
               hparams.decay_factor, warmup_steps, warmup_factor,
               (hparams.learning_rate * warmup_factor ** warmup_steps)))
        self.global_step = tf.Variable(0, trainable=False)

        params = tf.trainable_variables()
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            inv_decay = warmup_factor ** (
                tf.to_float(warmup_steps - self.global_step))
            self.learning_rate = tf.cond(
                self.global_step < hparams.learning_rate_warmup_steps,
                lambda: inv_decay * self.learning_rate,
                lambda: self.learning_rate,
                name="learning_rate_decay_warump_cond")

            if self.hparams.optimizer == "sgd":
                self.learning_rate = tf.cond(
                    self.global_step < hparams.start_decay_step,
                    lambda: self.learning_rate,
                    lambda: tf.train.exponential_decay(
                        self.learning_rate,
                        (self.global_step - hparams.start_decay_step),
                        hparams.decay_steps,
                        hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif hparams.optimizer == "adam":
                assert float(
                    hparams.learning_rate
                ) <= 0.001, "! High Adam learning rate %g" % hparams.learning_rate
                opt = tf.train.AdamOptimizer(self.learning_rate)

            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_gradients, gradient_norm_summary = Helpers.gradient_clip(
                gradients, max_gradient_norm=hparams.max_gradient_norm)

            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                                                      tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("train_loss", self.train_loss),
                                                  ] + gradient_norm_summary)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_summary = self._get_infer_summary(hparams)

        self.saver = tf.train.Saver(tf.global_variables())

        # Print trainable variables
        utils.print_out("# Trainable variables")
        for param in params:
            utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                              param.op.device))

    def init_embeddings(self, scope):
        """Init embeddings."""
        self.embedding_encoder, self.embedding_decoder = (
            Helpers.create_emb_for_encoder_and_decoder(
                share_vocab=self.hparams.share_vocab,
                src_vocab_size=self.hparams.src_vocab_size,
                tgt_vocab_size=self.hparams.tgt_vocab_size,
                src_embed_size=self.hparams.num_units,
                tgt_embed_size=self.hparams.num_units,
                num_partitions=self.hparams.num_embeddings_partitions,
                scope=scope))

    def build_graph(self, scope=None):
        utils.print_out("# creating %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            encoder_outputs, encoder_state = self._build_encoder()

            logits, sample_id, final_context_state = \
                self._build_decoder(encoder_outputs, encoder_state)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(Helpers.get_device_str(self.hparams.num_layers - 1, self.hparams.num_gpus)):
                    loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

    @abc.abstractmethod
    def _build_encoder(self):
        """Subclass must implement this.
        Build and run an RNN encoder.
        Args:
          hparams: Hyperparameters configurations.
        Returns:
          A tuple of encoder_outputs and encoder_state.
        """
        pass

    def _build_encoder_cell(self, num_layers, num_residual_layers,
                            base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return Helpers.create_rnn_cell(
            unit_type=self.hparams.unit_type,
            num_units=self.hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=self.hparams.forget_bias,
            dropout=self.hparams.dropout,
            num_gpus=self.hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def _build_decoder(self, encoder_outputs, encoder_state):
        tgt_sos_id = tf.cast(self.target_vocab.lookup(tf.constant(self.hparams.sos)),
                             tf.int32)
        tgt_eos_id = tf.cast(self.target_vocab.lookup(tf.constant(self.hparams.eos)),
                             tf.int32)

        if self.hparams.tgt_max_len_infer:
            maximum_iterations = self.hparams.tgt_max_len_infer
            utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(self.iterator.source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))

        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                encoder_outputs, encoder_state,
                self.iterator.source_sequence_length)

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                target_input = self.iterator.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder,
                                                         target_input)

                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,
                                                           self.iterator.target_sequence_length
                                                           , time_major=self.time_major)

                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                    output_time_major=self.time_major,
                                                                                    swap_memory=True,
                                                                                    scope=decoder_scope)

                sample_id = outputs.sample_id

                device_id = self.hparams.num_layers if self.hparams.num_layers < self.hparams.num_gpus else (
                self.hparams.num_layers - 1)
                with tf.device(Helpers.get_device_str(device_id, self.hparams.num_gpus)):
                    logits = self.output_layer(outputs.rnn_output)
            else:
                beam_width = self.hparams.beam_width
                length_penalty_weight = self.hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight
                    )
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        self.embedding_decoder, start_tokens, end_token
                    )
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state,
                                                                 output_layer=self.output_layer)

                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder, maximum_iterations=maximum_iterations,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope
                )

                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    @abc.abstractmethod
    def _build_decoder_cell(self, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Subclass must implement this.
        Args:
          hparams: Hyperparameters configurations.
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          source_sequence_length: sequence length of encoder_outputs.
        Returns:
          A tuple of a multi-layer RNN cell used by decoder
            and the intial state of the decoder RNN.
        """

    pass


    def decode(self, sess):
        """Decode a batch.
        Args:
          sess: tensorflow session to use.
        Returns:
          A tuple consiting of outputs, infer_summary.
            outputs: of size [batch_size, time]
        """
        _, infer_summary, _, sample_words = self.infer(sess)

        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            sample_words = sample_words.transpose()
        return sample_words, infer_summary


    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size])


    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.predict_count,
                         self.batch_size])


    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([
            self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
        ])


    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


    def _get_infer_summary(self, hparams):
        return tf.no_op()


    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.iterator.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        return loss
