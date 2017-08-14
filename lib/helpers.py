import os
import fnmatch
import numpy as np
from glob import glob

def gather_filepaths(directory, pattern="*"):
    paths = []
    for path, x, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepath = "%s/%s" % (path, filename)
            paths.append(filepath)
    return np.asarray(paths)

def gather_folders(root):
    return np.asarray(glob(os.path.join(root, "*")))
