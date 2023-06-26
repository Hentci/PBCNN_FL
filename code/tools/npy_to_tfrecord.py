# import tensorflow as tf
# import numpy as np
# import os


# def create_tf_example(feature_data, label):
#     # label = os.path.splitext(label)[0]  # Remove file extension from label
#     # label = int(label)  # Convert label to integer

#     # Convert the feature_data array to float32 data type
#     feature_data = feature_data.astype(np.float32)

#     # Create a feature dictionary
#     feature_dict = {
#         'data': tf.train.Feature(float_list=tf.train.FloatList(value=feature_data.flatten())),
#         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
#     }
    
#     # Create an example from the feature dictionary
#     example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
#     return example


# def convert_to_tfrecord(npy_files, output_file):
#     with tf.io.TFRecordWriter(output_file) as writer:
#         cnt = 0
#         for npy_file in npy_files:
#             # Load the data and label from the .npy file
#             data = np.load(npy_file)
#             label = npy_file.split('.')[0]  # Extract the label from the file name
#             label = label[label.rfind('/') + 1:]
#             print(label)
#             label = cnt
#             cnt += 1

#             # Create the TFRecord example
#             example = create_tf_example(data, label)
            
#             # Serialize and write the example to the TFRecord file
#             writer.write(example.SerializeToString())

# a = ['train', 'test', 'valid']


# for ele in a:

#     # Example usage
#     label_names = ['bruteforce-ftp', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http',
#                     'dos-hulk', 'botnet', 'bruteforce-ssh', 'dos-slowhttptest',
#                     'webattack', 'dos-slowloris', 'benign']

#     folder_path = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/' + ele
#     npy_files = [os.path.join(folder_path, label_name + '.npy') for label_name in label_names]

#     output_file = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/to_tfrecord/'+ ele +'/part-000.tfrecord'

#     convert_to_tfrecord(npy_files, output_file)


import tensorflow as tf
import numpy as np
import os


def create_tf_example(feature_data, label):
    feature_data = feature_data.astype(np.float32)

    feature_dict = {
        'data': tf.train.Feature(float_list=tf.train.FloatList(value=feature_data.flatten())),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


def convert_to_tfrecord(npy_files, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        cnt = 0
        for npy_file in npy_files:
            data = np.load(npy_file)
            label = npy_file.split('.')[0]
            label = label[label.rfind('/') + 1:]
            print(label)
            label = cnt
            cnt += 1

            example = create_tf_example(data, label)

            writer.write(example.SerializeToString())


a = ['train', 'test', 'valid']

for ele in a:
    label_names = ['bruteforce-ftp', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http',
                   'dos-hulk', 'botnet', 'bruteforce-ssh', 'dos-slowhttptest',
                   'webattack', 'dos-slowloris', 'benign']

    folder_path = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/' + ele
    npy_files = [os.path.join(folder_path, label_name + '.npy') for label_name in label_names]

    output_folder = '/trainingData/sage/CIC-IDS2018-byte/CIC-IDS-2018/to_tfrecord/' + ele
    os.makedirs(output_folder, exist_ok=True)

    # Shuffle the data before converting to TFRecord
    np.random.shuffle(npy_files)

    num_records_per_file = 2  # Number of records per TFRecord file
    num_files = int(np.ceil(len(npy_files) / num_records_per_file))

    for i in range(num_files):
        output_file = os.path.join(output_folder, f'part-{i:03d}.tfrecord')
        start_idx = i * num_records_per_file
        end_idx = min((i + 1) * num_records_per_file, len(npy_files))
        convert_to_tfrecord(npy_files[start_idx:end_idx], output_file)
