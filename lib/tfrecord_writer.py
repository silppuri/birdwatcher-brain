import os
import sys
import pdb
import librosa
import tensorflow as tf
import numpy as np
from helpers import gather_filepaths
from helpers import gather_folders, gather_filepaths
from sklearn.model_selection import train_test_split
from shutil import copy2
from glob import glob

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

SEED = 42
classes = [path.split("/")[-1] for path in glob(os.path.join("data/mp3", "*"))]
classes.sort()
print(classes)
np.save("data/classes.npy", classes)

three_seconds = 44100 * 3
two_seconds = 44100 * 2

def chunks(l, n, step_size):
    """Yield successive n-sized chunks from l with step size."""
    for i in range(0, len(l), step_size):
        yield l[i:i + n]

def remove_silence(src, dst):
    for folder in gather_folders(src):
        filepaths = gather_filepaths(folder, pattern="*.mp3")
        for index, filepath in enumerate(filepaths):
            species = filepath.split("/")[-2]
            destination = os.path.join(dst, species)
            if not os.path.exists(destination):
                os.makedirs(destination)

            file_destination = destination + "/" + species + str(index) + ".wav"
            os.system("sox -v 0.99 " + filepath + " " + file_destination + " rate 44100 silence -l 1 0.3 1% -1 2.0 1")

def remove_short(src):
    filepaths = gather_filepaths(src, pattern="*.wav")
    for filename in filepaths:
        if librosa.get_duration(filename=filename) < 3.0:
            print("removing %s" % filename)
            os.remove(filename)

def generate_more_samples(src, dst):
    files = gather_filepaths(src, pattern="*.wav")
    for i in range(len(files)):
        audio, _ = librosa.load(files[i], sr=44100)
        num_for_one_label = 0
        for j, three_second_chunk in enumerate(chunks(audio, three_seconds, round(44100 * 1.5))):
            if len(three_second_chunk) < three_seconds: continue
            label = classes.index(files[i].split("/")[-2])
            destination = os.path.join(dst, str(label))
            if not os.path.exists(destination):
                os.makedirs(destination)
            librosa.output.write_wav(os.path.join(destination, str(j) + '.wav'), three_second_chunk, 44100)

def copy_file(src, destination, filename):
    if not os.path.exists(destination):
        os.makedirs(destination)
    new_file = os.path.join(destination, str(filename) + ".wav")
    copy2(src, new_file)

def save_files(files, dst):
    for idx, filepath in enumerate(files):
        target = filepath.split("/")[-2]
        destination = os.path.join(dst, target)
        copy_file(filepath, destination, idx)

def data_split(src, train_dst='data/train', test_dst='data/test'):
    for folder in gather_folders(src):
        filepaths = gather_filepaths(folder, pattern="*.wav")
        train_set, test_set = train_test_split(filepaths, test_size=0.2, random_state=SEED)
        print("train set with %d samples", len(train_set))
        print("test set with %d samples", len(test_set))
        save_files(train_set, train_dst)
        save_files(test_set, test_dst)

def write_tfrecord(src, dst=None):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(dst)
    files = gather_filepaths(src, pattern="*.wav")
    for i in range(len(files)):
        audio, _ = librosa.load(files[i], sr=44100)
        label = int(files[i].split("/")[-2])
        # Create a feature
        feature = {'label': _int64_feature(label),
                   'audio': _bytes_feature(tf.compat.as_bytes(audio.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()

if __name__ == '__main__':
    remove_silence('data/mp3', 'data/audio')
    remove_short('data/audio')
    generate_more_samples('data/audio', 'data/tmp')
    data_split('data/tmp', train_dst='data/train', test_dst='data/test')
    write_tfrecord('data/train', dst='data/train.tfrecord')
    write_tfrecord('data/test', dst='data/test.tfrecord')
