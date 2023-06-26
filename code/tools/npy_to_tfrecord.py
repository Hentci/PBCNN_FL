import tensorflow as tf
import numpy as np
import os

def create_tf_example(feature_data, label):
    # Convert the feature_data array to float32 data type
    feature_data = feature_data.astype(np.float32)

    # Create a feature dictionary
    feature_dict = {
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=feature_data.flatten())),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()]))
    }
    
    # Create an example from the feature dictionary
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return example



def convert_to_tfrecord(npy_files, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for npy_file in npy_files:
            # Load the data and label from the .npy file
            data = np.load(npy_file)
            label = npy_file.split('.')[0]  # Extract the label from the file name
            label = label[label.rfind('/') + 1:]
            print(label)
            
            # Create the TFRecord example
            example = create_tf_example(data, label)
            
            # Serialize and write the example to the TFRecord file
            writer.write(example.SerializeToString())

# Example usage
label_names = ['bruteforce-ftp', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http',
                'dos-hulk', 'botnet', 'bruteforce-ssh', 'dos-slowhttptest',
                'webattack', 'dos-slowloris', 'benign']

folder_path = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/valid'
npy_files = [os.path.join(folder_path, label_name + '.npy') for label_name in label_names]

output_file = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/to_tfrecord/valid/part-000.tfrecord'

convert_to_tfrecord(npy_files, output_file)
