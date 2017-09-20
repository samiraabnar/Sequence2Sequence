import tensorflow as tf
import BaseModel, Helpers

from utils import iterator_utils
from utils import misc_utils as utils


class Model(BaseModel.BaseModel):

    def _build_encoder(self):
        source = self.iterator.source
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as encoder_scope:
            dtype = encoder_scope.dtype

            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, source)

            if self.hparams.encoder_type == "uni":
                utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                                (self.hparams.num_layers, self.hparams.num_residual_layers))
                cell = self._build_encoder_cell(
                    self.hparams.num_layers, self.hparams.num_layers, self.hparams.num_residual_layers)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=self.iterator.source_sequence_length,
                    time_major=self.time_major)
            elif self.hparams.encoder_type == "bi":
                num_bi_layers = int(self.hparams.num_layers / 2)
                num_bi_residual_layers = int(self.hparams.num_residual_layers / 2)
                utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                                (num_bi_layers, num_bi_residual_layers))

                encoder_outputs, bi_encoder_state = (
                    self._build_bi_rnn(
                        inputs=encoder_emb_inp,
                        sequence_length=self.iterator.source_sequence_length,
                        dtype=dtype,
                        hparams=self.hparams,
                        num_bi_layers=num_bi_layers,
                        num_bi_residual_layers=num_bi_residual_layers))

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

            return encoder_outputs, encoder_state

    def _build_bi_rnn(self,inputs,sequence_length,dtype,num_bi_layers, num_bi_residual_layers,base_gpu=0):
        forward_cell = self._build_encoder_cell(num_bi_layers,num_bi_residual_layers,base_gpu=base_gpu)
        backward_cell = self._build_decoder_cell(num_bi_layers,num_bi_residual_layers,base_gpu=(base_gpu + num_bi_layers))

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell,backward_cell,inputs, dtype=dtype, sequence_length=sequence_length,
            time_major=self.time_major)

        return tf.concat(bi_outputs,-1), bi_state

    def _build_decoder_cell(self,encoder_outputs,encoder_state, sequence_length):
        if self.hparams.attention:
            raise ValueError("Basic Model doesn't support attention.")

        cell = Helpers.create_rnn_cell(
            unit_type=self.hparams.unit_type,
            num_units=self.hparams.num_units,
            num_layers=self.hparams.num_layers,
            num_residual_layers=self.hparams.num_residual_layers,
            forget_bias=self.hparams.forget_bias,
            dropout=self.hparams.dropout,
            num_gpus=self.hparams.num_gpus,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn
        )

        if self.mode == tf.contrib.learn.ModeKeys.INFER and self.hparams.beam_width > 0:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.hparams.beam_width)
        else:
            decoder_initial_state = encoder_state

        return cell,decoder_initial_state

