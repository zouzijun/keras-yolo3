import yaml
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_body, tiny_yolo_body


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

class converter(object):
    def __init__(self, params):
        self.model_path = '../model_data/yolo_weights.h5'
        self.class_cnt = 20
        self.anchor_cnt = 9
        self.sess = K.get_session()
        self._generate_model()

    def _generate_model(self):
        # Load model, or construct model and load weights.
        is_tiny_version = self.anchor_cnt==6 # default setting
        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except:
            if is_tiny_version:
                self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), self.anchor_cnt//2, self.class_cnt)
            else:
                self.yolo_model = yolo_body(Input(shape=(None,None,3)), self.anchor_cnt//3, self.class_cnt)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == self.anchor_cnt/len(self.yolo_model.output) * (self.class_cnt + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

    def save_pb_file(self):
        frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in self.yolo_model.outputs])
        tf.train.write_graph(frozen_graph, "./", "yolo_model.pb", as_text=False)



if __name__ == '__main__':
    # Read config file
    with open('../config.yml') as f:
        params = yaml.load(f)
        conv = converter(params)
        conv.save_pb_file()