import scipy
import os
import librosa

from shutil import copy2
from helpers import gather_folders, gather_filepaths
from sklearn.model_selection import train_test_split

SEED = 42
RAW_MP3S = "data/mp3"
DESTINATION = "data"
TMP_FILE = "tmp.wav"
AUDIO_FILES = "data/audio"

def copy_file(src, destination, filename):
    if not os.path.exists(destination):
        os.makedirs(destination)
    new_file = os.path.join(destination, str(filename) + ".wav")
    copy2(src, new_file)

def save_files(files, set_type):
    for idx, filepath in enumerate(files):
        target = filepath.split("/")[-2]
        destination = os.path.join(DESTINATION, set_type, target)
        copy_file(filepath, destination, idx)

def split_files_and_resample():
    for folder in gather_folders(RAW_MP3S):
        filepaths = gather_filepaths(folder, pattern="*.mp3")
        for index, filepath in enumerate(filepaths):
            species = filepath.split("/")[-2]
            destination = os.path.join(AUDIO_FILES, species)
            if not os.path.exists(destination):
                os.makedirs(destination)

            os.system("sox -v 0.99 " + filepath + " " + TMP_FILE + " rate 44100 silence -l 1 0.1 1% -1 0.4 1%")
            file_destination = destination + "/" + species + str(index) + ".wav"
            os.system("sox " + TMP_FILE + " " + file_destination + " trim 0 3 : newfile : restart")

def remove_short():
    filepaths = gather_filepaths(AUDIO_FILES, pattern="*.wav")
    for filename in filepaths:
        if librosa.get_duration(filename=filename) < 3.0:
            print("removing %s" % filename)
            os.remove(filename)

def data_split():
    for folder in gather_folders(AUDIO_FILES):
        filepaths = gather_filepaths(folder, pattern="*.wav")
        train_set, test_set = train_test_split(filepaths, test_size=0.2, random_state=SEED)
        print("train set with %d samples", len(train_set))
        print("test set with %d samples", len(test_set))
        save_files(train_set, "train")
        save_files(test_set, "test")

# split_files_and_resample()
# remove_short()
data_split()
