import pdb
import tensorflow as tf
from birdwatcher.generators import compose, stft, amplitude_to_db, read_audio, reshape

AUDIO_SHAPE = (44100*3, 1)
clean_samples = compose(reshape, amplitude_to_db, stft, read_audio)

x = tf.placeholder(tf.float32, shape=AUDIO_SHAPE)
out = clean_samples(x)

sess = tf.Session()
tf.train.write_graph(sess.graph_def, 'models', 'stft.pbtxt')
