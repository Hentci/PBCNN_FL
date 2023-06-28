import tensorflow as tf

# Define the path to your TFRecord file
# tfrecord_file = '/trainingData/sage/PBCNN/data/demo_tfrecord/part-000.tfrecord'
tfrecord_file = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/to_tfrecord/test/part-000.tfrecord'

# Create a TFRecordDataset object to read the file
dataset = tf.data.TFRecordDataset(tfrecord_file)

# Iterate over the records in the dataset
for record in dataset:
    # Parse each record
    example = tf.train.Example()
    example.ParseFromString(record.numpy())

    # Print the dimensions of the features
    features = example.features.feature
    for feature_name, feature in features.items():
        feature_length = len(feature.float_list.value)  # Adjust based on the feature type
        print("Feature:", feature_name)
        print("Length:", feature_length)
    
    print("----------------------")
