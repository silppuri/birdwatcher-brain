import os
import fnmatch
import librosa
import threading
import random
import tensorflow as tf
import numpy as np
import pdb

FILE_PATTERN = '*.wav'


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def gather_paths(directory):
    paths = []
    for path, _, files in os.walk(directory):
        for filename in fnmatch.filter(files, FILE_PATTERN):
            filepath = "%s/%s" % (path, filename)
            paths.append(filepath)
    return np.asarray(paths)


def audio_loader(directory, sample_rate, labels, files):
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        label = filename.split("/")[2]
        label_index = labels.index(label)
        one_hot_label = np.eye(len(labels))[label_index]
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        X = librosa.stft(audio, n_fft=512, hop_length=128, win_length=512)
        X = librosa.core.amplitude_to_db(X, ref=np.max)
        yield np.reshape(X[:256, :512], 256 * 512), one_hot_label


def get_labels(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class AudioReader():
    """
    Background audio file reader and preprocessor that feeds the files
    for Tensorflow queue
    """

    def __init__(self, audio_dir, coord, sample_rate=32000, queue_size=32):
        self._audio_dir = audio_dir
        self._coord = coord
        self._sample_rate = sample_rate
        self._queue_size = queue_size
        self._labels = list(get_labels(audio_dir))

        self._threads = []
        self._image_placeholder = tf.placeholder(dtype=tf.float32,
                shape=None)
        self._label_placeholder = tf.placeholder(dtype=tf.int64,
                shape=None)
        self._queue = tf.PaddingFIFOQueue(
                queue_size,
                ['float32', 'int64'],
                shapes=[[256 * 512], [len(self._labels)]])
        self._enqueue = self._queue.enqueue(
            [self._image_placeholder, self._label_placeholder])
        self._files = gather_paths(self._audio_dir)
        print("files length: {}".format(len(self._files)))

    def filecount(self):
        return len(self._files)

    def dequeue(self, num_elements):
        return self._queue.dequeue_many(num_elements)

    def main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = audio_loader(
                self._audio_dir, self._sample_rate, self._labels, self._files)
            for image, label in iterator:
                if self._coord.should_stop():
                    stop = True
                    break
                sess.run(self._enqueue,
                         feed_dict={self._image_placeholder: image,
                                    self._label_placeholder: label})

    def start_threads(self, sess, thread_count=1):
        for _ in range(thread_count):
            thread = threading.Thread(target=self.main, args=(sess,))
            thread.daemon = True
            thread.start()
            self._threads.append(thread)
        return self._threads
