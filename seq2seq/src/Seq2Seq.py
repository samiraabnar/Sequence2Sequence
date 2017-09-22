from Model import *

class Seq2Seg(object):

    def __init__(self):
        pass




def train(hparams, scope=None, target_session=""):

    model_creator = Model
    train_model = Helpers.create_train_model(model_creator, hparams, scope)
    eval_model = Helpers.create_eval_model(model_creator, hparams, scope)
    infer_model = Helpers.create_infer_model(model_creator, hparams, scope)

    train_sess = tf.Session(
        target=target_session, config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(
        target=target_session, config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(
        target=target_session, config=config_proto, graph=infer_model.graph)

if __name__ == '__main__':

        