import pdb
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import functools
from glob import glob
from tensorflow.contrib.data import TFRecordDataset
from tensorflow.contrib.ffmpeg import decode_audio


NUM_CLASSES = len(np.load('data/classes.npy'))

identity = lambda x: x

def amplitude_to_db(S):
    magnitude = tf.abs(S)
    ref_value = tf.pow(tf.reduce_max(magnitude), 2)
    magnitude = tf.pow(magnitude, 2)
    log_spec = tf.multiply(10.0, tf.log(tf.maximum(1e-10, magnitude)))
    log_spec = tf.subtract(log_spec, tf.multiply(10.0, tf.log(tf.maximum(1e-10, ref_value))))
    return log_spec

def audio_paths_and_labels(folder, pattern="*.wav"):
    classes = [path.split("/")[-1] for path in glob(os.path.join(folder, "*"))]
    filenames = glob(os.path.join(folder, "**", pattern))
    labels = [classes.index(filename.split("/")[-2]) for filename in filenames]
    return np.asarray(filenames), np.asarray(labels)

def read_audio(audio):
    return tf.reshape(audio, [1, 44100 * 3]) # reshape from shape (44100*3, 1) to (44100 * 3)


def stft(audio):
    return tf.transpose(tf.contrib.signal.stft(audio, 512, 256))

def normalize_image(X):
    X = tf.image.per_image_standardization(X)
    return tf.nn.relu(X)

def reshape(X):
    return tf.reshape(X, [257, 515])

def noise(X):
    return tf.add(X, tf.random_normal(X.shape, mean=0.0, stddev=0.3))

def train_sample(X):
    return X, X

def default_parser(filename):
    audio = read_audio(filename)
    X = stft(audio)
    X = amplitude_to_db(X)
    X = normalize_image(X)
    X_noisy = add_noise(X)
    return X_noisy, X

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

class Generator():
    def __init__(self, tf_record_path, batch_size=32, num_epochs=10, parser=identity):
        self._filename = tf_record_path
        self._batch_size = batch_size
        self._parser = parser
        dataset = TFRecordDataset(self._filename)
        dataset = dataset.map(self._parse_tfrecord, num_threads=4, output_buffer_size=2*4*batch_size)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.batch(batch_size)
        self._session = tf.Session()
        self._iterator = dataset.make_one_shot_iterator()
        self._next_element = self._iterator.get_next()

    def _parse_tfrecord(self, example_proto):
        features = {"audio": tf.FixedLenFeature([], tf.string),
                    "label": tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        audio = tf.decode_raw(parsed_features['audio'], tf.float32)
        label = tf.cast(parsed_features['label'], tf.int32)
        X = self._parser(audio)
        return X, tf.one_hot(label, NUM_CLASSES, axis=-1)

    def next_batch(self):
        while 1:
            audios, labels = self._session.run(self._next_element)
            yield (audios, labels)
