# BIRDWATHCER

Bird identification app

## Data

Steps:

1. Remove samples shorter than 3sec,
2. Remove silences,
2. Split samples into 3sec pieces, with a sliding window of 1.5sec (this generates more samples)
3. Split data into three datasets,
4. Forget about test data and work only with train and validation data,
4. Write tfrecords
