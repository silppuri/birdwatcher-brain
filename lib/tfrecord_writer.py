import os
import sys
import pdb
import librosa
import tensorflow as tf
import numpy as np
from helpers import gather_filepaths
from glob import glob

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


classes = [path.split("/")[-1] for path in glob(os.path.join("data/train", "*"))]
classes.sort()
np.save("data/classes.npy", classes)

def convert(folder):
    train_filename = folder + '.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    files = gather_filepaths(folder, pattern="*.wav")
    for i in range(len(files)):
        # print how many vaws are saved every 1000 wavs
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(files)))
            sys.stdout.flush()
        # Load the image
        audio, _ = librosa.load(files[i], sr=44100)
        assert audio.shape[0] == 44100 * 3, "file %s is corrupted" % files[i]
        label = classes.index(files[i].split("/")[-2])
        # Create a feature
        feature = {'label': _int64_feature(label),
                   'audio': _bytes_feature(tf.compat.as_bytes(audio.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

convert('data/test')
convert('data/train')
