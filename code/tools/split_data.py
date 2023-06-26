import numpy as np
import os
import random

# Set the seed for reproducibility
random.seed(42)

# Load the data from the .npy file
data = np.load('/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/11 class/webattack.npy')

# Shuffle the data
random.shuffle(data)

# Calculate the indices for train, valid, and test sets
total_samples = data.shape[0]
train_samples = int(total_samples * 0.6)
valid_samples = int(total_samples * 0.1)
test_samples = total_samples - train_samples - valid_samples

# Split the data into train, valid, and test sets
train_data = data[:train_samples]
valid_data = data[train_samples:train_samples + valid_samples]
test_data = data[train_samples + valid_samples:]

# Save the split data as .npy files
np.save('/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/train/webattack.npy', train_data)
np.save('/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/valid/webattack.npy', valid_data)
np.save('/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/test/webattack.npy', test_data)
