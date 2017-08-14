import pdb
import os
from keras.models import model_from_json
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

sess = tf.Session()
K.set_session(sess)

K.set_learning_phase(0)  # all new operations will be in test mode from now on

# serialize the model and get its weights, for quick re-building

previous_model = load_model('models/birdwatcher.h5')
previous_model.save_weights('models/birdwatcher_weights.h5')
model_json = previous_model.to_json()
new_model = model_from_json(model_json)
new_model.load_weights('models/birdwatcher_weights.h5')
checkpoint_prefix = os.path.join("models", "saved_checkpoint")


export_path = 'models/'
export_version = 1

checkpoint_state_name = "checkpoint_state"
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=new_model.input,
                                              scores_tensor=new_model.output)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(export_version), sess)
tf.train.write_graph(sess.graph.as_graph_def(), 'models/', 'birdwatcher.pbtxt')
checkpoint_path = saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)

input_graph_name = "birdwatcher.pbtxt"
output_graph_name = "frozen.pb"

input_graph_path = os.path.join("models", input_graph_name)
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = os.path.join("models", 'saved_checkpoint') + "-0"

# Note that we this normally should be only "output_node"!!!
output_node_names = "loss/Softmax"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = os.path.join("models", output_graph_name)
clear_devices = False
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, input_checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name, output_graph_path,
                          clear_devices, False)
# python env/lib/python3.6/site-packages/tensorflow/python/tools/optimize_for_inference.py --input=models/frozen.pb --output=models/inference.pb --input_names=input_1 --output_names=loss/Softmax
