import numpy as np

with open('data/labels.txt', 'w') as labels_file:
  classes = np.load('data/classes.npy')
  for klass in classes:
    labels_file.write(klass)
    labels_file.write('\n')
