import os
import pickle
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from absl import logging, app
from sklearn.metrics import classification_report
from tensorflow import keras as K
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer
from tqdm import tqdm

MAX_PKT_BYTES = 50 * 50
MAX_PKT_NUM = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE


def plot_heatmap(report, y_labels=None):
    mt = []
    if y_labels is None:
        y_labels = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'sql-injection',
                    'dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss', 'dos-slowhttptest',
                    'bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp', 'infiltration']
    support = []
    x_labels = ['precision', 'recall', 'f1-score']
    for name in y_labels:
        mt.append([
            report[name]['precision'],
            report[name]['recall'],
            report[name]['f1-score']
        ])
        support.append(report[name]['support'])
    assert len(support) == len(y_labels)
    y_labels_ = []
    for i in range(len(y_labels)):
        y_labels_.append(f'{y_labels[i]} ({support[i]})')
    plt.figure(figsize=(5, 6), dpi=200)
    sns.set()
    sns.heatmap(mt, annot=True, xticklabels=x_labels, yticklabels=y_labels_, fmt='.4f',
                linewidths=0.5, cmap='PuBu', robust=True)
    plt.show()

class Client():
    def __init__(self):
        # 這邊是底下 TF 的 _init_() 來的
        self.optimizer = K.optimizers.Adam()
        self.loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
        self.acc_func = K.metrics.sparse_categorical_accuracy

        self.train_losses = []
        self.valid_losses = []
        self.train_acc = []
        self.valid_acc = []

        self.total_loss = 0.
        self.total_match = 0
    
    def reset(self):
        self.sample_count = 0
        self.total_loss = 0.
        self.total_match = 0
    
    def train_step(self, features, labels):
        # 來自底下 TF 的 _train_step()
        with tf.GradientTape() as tape:
            y_predict = self.model(features, training=True)
            loss = self.loss_func(labels, y_predict)
            acc_match = self.acc_func(labels, y_predict)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy().sum(), acc_match.numpy().sum()

class TF(object):

    def __init__(self, pkt_bytes, pkt_num, model,
                 train_path, valid_path, test_path,
                 batch_size=128, num_class=15):
        model = model.lower().strip()
        assert pkt_bytes <= MAX_PKT_BYTES, f'Check pkt bytes less than max pkt bytes {MAX_PKT_BYTES}'
        assert pkt_num <= MAX_PKT_NUM, f'Check pkt num less than max pkt num {MAX_PKT_NUM}'
        assert model in ('pbcnn', 'en_pbcnn'), f'Check model type'

        self._pkt_bytes = pkt_bytes
        self._pkt_num = pkt_num
        self._model_type = model

        assert os.path.isdir(train_path)
        assert os.path.isdir(valid_path)
        assert os.path.isdir(test_path)

        self._train_path = train_path
        self._valid_path = valid_path
        self._test_path = test_path

        self._batch_size = batch_size
        self._num_class = num_class

        self._prefix = f'bytes_{pkt_bytes}_num_{pkt_num}_{model}'
        if not os.path.exists(self._prefix):
            os.makedirs(self._prefix)
        
        # local epoch 次數
        self.local_epochs = 1
        # 建立 clients
        self.clients = []
        self.client_num = 10
        for i in range(self.client_num):
            self.clients.append(Client())

    def __new__(cls, *args, **kwargs):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logging.set_verbosity(logging.INFO)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

        tf.debugging.set_log_device_placement(False)
        tf.config.set_soft_device_placement(True)
        # tf.config.threading.set_inter_op_parallelism_threads(0)
        # tf.config.threading.set_intra_op_parallelism_threads(0)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
        return super().__new__(cls)

    def _parse_sparse_example(self, example_proto):
        features = {
            'sparse': tf.io.SparseFeature(index_key=['idx1', 'idx2'],
                                        value_key='val',
                                        dtype=tf.int64,
                                        size=[MAX_PKT_NUM, MAX_PKT_BYTES]),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
            'byte_len': tf.io.FixedLenFeature([], dtype=tf.int64),
            'last_time': tf.io.FixedLenFeature([], dtype=tf.float32),
        }
        batch_sample = tf.io.parse_example(example_proto, features)
        sparse_features = batch_sample['sparse']
        labels = batch_sample['label']
        sparse_features = tf.sparse.slice(sparse_features, start=[0, 0], size=[self._pkt_num, 60])
        dense_features = tf.sparse.to_dense(sparse_features)
        dense_features = tf.cast(dense_features, tf.float32) / 255.
        return dense_features, labels


    def _generate_ds(self, path_dir, use_cache=False):
        assert os.path.isdir(path_dir)
        ds = tf.data.Dataset.list_files(os.path.join(path_dir, '*.tfrecord'), shuffle=True)
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x).map(self._parse_sparse_example),
            cycle_length=AUTOTUNE,
            block_length=8,
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.batch(self._batch_size, drop_remainder=False)
        if use_cache:
            ds = ds.cache()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def _init_input_ds(self):
        self._train_ds = self._generate_ds(self._train_path)
        print('train ds size: ', len(list(self._train_ds)))
        self._valid_ds = self._generate_ds(self._valid_path)
        print('valid ds size: ', len(list(self._valid_ds)))
        
        # Use tqdm to create a progress bar
        progress_bar = tqdm(total=self.client_num, desc="Initializing Input DS", unit="client")
        
        # 分 data
        spilt_ds = self._train_ds.shuffle(len(list(self._train_ds)), reshuffle_each_iteration=False)
        # 算每個 client 拿幾筆 data
        data_n = int((1 / self.client_num) * len(list(spilt_ds)))
        
        for i in range(self.client_num):
            temp = spilt_ds.take(data_n)
            spilt_ds = spilt_ds.skip(data_n)
            self.clients[i].ds = temp
            self.clients[i].sample_count = data_n
            
            # Update the tqdm progress bar
            progress_bar.update(1)
        
        # Close the tqdm progress bar
        progress_bar.close()
        
        # 確認分完的結果
        # for i in range(self.client_num):
        #     print('len_client[{}]_ds: {}'.format(i, len(list(self.clients[i].ds))))
        #     print('client[{}]_ds[0]: {}'.format(i, list(self.clients[i].ds.as_numpy_iterator())[0]))

    @staticmethod
    def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
        x = layers.Conv2D(filters=filters, kernel_size=(height, width),
                          strides=1, data_format=data_format)(x)
        x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
        x = layers.Activation(activation='relu')(x)
        x = tf.reduce_max(x, axis=1, keepdims=False)
        return x

    @staticmethod
    def _conv1d_block(x, filters, data_format='channels_last'):
        x = layers.Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
        x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def _enhanced_pbcnn(self):
        x = Input(shape=(self._pkt_num, self._pkt_bytes))
        y = tf.reshape(x, shape=(-1, self._pkt_bytes, 1))
        y = self._conv1d_block(y, filters=64)
        # y = self._conv1d_block(y, filters=64)
        y = layers.MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(y)
        y = self._conv1d_block(y, filters=128)
        # y = self._conv1d_block(y, filters=128)
        y = layers.MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(y)
        filters = 256
        y = self._conv1d_block(y, filters=filters)
        y = tf.reduce_max(y, axis=1, keepdims=False)
        y = tf.reshape(y, shape=(-1, self._pkt_num, filters, 1))
        y1 = self._text_cnn_block(y, filters=256, height=3, width=filters)
        y2 = self._text_cnn_block(y, filters=256, height=4, width=filters)
        y3 = self._text_cnn_block(y, filters=256, height=5, width=filters)
        y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)

        y = layers.Flatten()(y)
        y = layers.Dense(512, activation='relu')(y)
        y = layers.Dense(256, activation='relu')(y)
        # y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(self._num_class, activation='linear')(y)

        return Model(inputs=x, outputs=y)

    def _pbcnn(self):
        x = Input(shape=(self._pkt_num, self._pkt_bytes))
        y = tf.reshape(x, shape=(-1, self._pkt_num, self._pkt_bytes, 1))
        data_format = 'channels_last'
        y1 = self._text_cnn_block(y, filters=256, height=3, width=self._pkt_bytes)
        y2 = self._text_cnn_block(y, filters=256, height=4, width=self._pkt_bytes)
        y3 = self._text_cnn_block(y, filters=256, height=5, width=self._pkt_bytes)
        y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)
        y = layers.Flatten(data_format=data_format)(y)
        y = layers.Dense(512, activation='relu')(y)
        y = layers.Dense(256, activation='relu')(y)
        # y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(self._num_class, activation='linear')(y)
        return Model(inputs=x, outputs=y)

    def _init_model(self):
        if self._model_type == 'pbcnn':
            # 原本的拿來當 global model
            self._model = self._pbcnn()
            # 建立每個 client 的 model
            for i in range(self.client_num):
                self.clients[i].model = self._pbcnn()
        else:
            self._model = self._enhanced_pbcnn()
            for i in range(self.client_num):
                self.clients[i].model = self._enhanced_pbcnn()
        # self._model.summary()

    def predict(self, model_dir, data_dir=None, digits=6):
        # model = tf.saved_model.load()
        model = K.models.load_model(model_dir)
        if data_dir:
            test_ds = self._generate_ds(data_dir)
        else:
            test_ds = self._generate_ds(self._test_path)
        y_pred, y_true = [], []
        for features, labels in test_ds:
            y_ = model.predict(features)
            y_ = np.argmax(y_, axis=-1)
            y_pred.append(y_)
            y_true.append(labels.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        label_names = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'sql-injection',
                       'dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss', 'dos-slowhttptest',
                       'bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp', 'infiltration']

        cl_re = classification_report(y_true, y_pred, digits=digits,
                                      labels=[i for i in range(self._num_class)],
                                      target_names=label_names, output_dict=True)
        accuracy = round(cl_re['accuracy'], digits)
        precision = round(cl_re['macro avg']['precision'], digits)
        recall = round(cl_re['macro avg']['recall'], digits)
        f1_score = round(cl_re['macro avg']['f1-score'], digits)

        # print(f'Macro Avg')
        # print(f'Accuracy: \t{accuracy} \n'
        #       f'Precision: \t{precision} \n'
        #       f'Recall: \t{recall} \n'
        #       f'F1-Score: \t{f1_score}')
        # # plot_heatmap(cl_re, y_labels=label_names)
        # return y_true, y_pred, cl_re

        return accuracy, precision, recall, f1_score, cl_re

    def init(self):
        print('init 1')
        self._init_input_ds()
        print('init 2')
        self._init_model()
        print('init 3')

    def _init_(self):
        self._optimizer = K.optimizers.Adam()
        # self._loss_func = K.losses.sparse_categorical_crossentropy
        self._loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
        self._acc_func = K.metrics.sparse_categorical_accuracy

        self._train_losses = []
        self._valid_losses = []
        self._train_acc = []
        self._valid_acc = []

    def _train_step(self, features, labels):
        with tf.GradientTape() as tape:
            y_predict = self._model(features, training=True)
            loss = self._loss_func(labels, y_predict)
            acc_match = self._acc_func(labels, y_predict)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss.numpy().sum(), acc_match.numpy().sum()

    def _test_step(self, features, labels):
        y_predicts = self._model(features, training=False)
        loss = self._loss_func(labels, y_predicts)
        acc_match = self._acc_func(labels, y_predicts)
        return loss.numpy().sum(), acc_match.numpy().sum()
    
    def weight_scaling_factor(self, client):
        # 算 scaling factor
        global_count = len(list(self._train_ds))
        local_count = client.sample_count
        return local_count/global_count
    
    def scale_model_weights(self, weight, scalar):
        # 把 local 訓練後的 model weight 乘上 scaling factor
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final
    
    def sum_scaled_weights(self, scaled_weight_list):
        # 把所有 client scaling 後的 model weight 相加
        avg_grad = list()
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad

    def train(self,
              epochs,
              log_freq=10,
              valid_freq=1,
              model_dir=f'models_tf',
              history_path='train_history.pkl',
              DEBUG=False):
        history_path = os.path.join(self._prefix, history_path)
        model_dir = os.path.join(self._prefix, model_dir)

        self._init_()
        steps = 1

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs))

            # 紀錄目前 global model 的 weight
            global_weights = self._model.get_weights()
            scaled_local_weight_list = list()

            for i in range(self.client_num):
                print('client ', i)
                # 重製 total_loss 那些
                self.clients[i].reset()
                # 把 global model 分給這個 client
                self.clients[i].model.set_weights(global_weights)
                steps = 0

                for local_epoch in range(self.local_epochs):
                    # 訓練的方式跟原本的 PBCNN 相同
                    for features, labels in self.clients[i].ds:
                        loss, match = self.clients[i].train_step(features, labels)
                        self.clients[i].total_loss += loss
                        self.clients[i].sample_count += len(features)
                        avg_train_loss = self.clients[i].total_loss / self.clients[i].sample_count
                        self.clients[i].train_losses.append(avg_train_loss)

                        self.clients[i].total_match += match
                        avg_train_acc = self.clients[i].total_match / self.clients[i].sample_count
                        self.clients[i].train_acc.append(avg_train_acc)
                        steps += 1
                
                print('Epoch %d, step %d, avg loss %.6f, avg acc %.6f' % (epoch, steps, avg_train_loss, avg_train_acc))

                # 開始做 FedAvg
                scaling_factor = self.weight_scaling_factor(self.clients[i])
                # print('scaling factor: ', scaling_factor)
                scaled_weights = self.scale_model_weights(self.clients[i].model.get_weights(), scaling_factor)
                scaled_local_weight_list.append(scaled_weights)

            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            # 更新 global model
            self._model.set_weights(average_weights)

            # 驗證 global model (方法跟原本的 PBCNN 相同)
            if valid_freq > 0 and epoch % valid_freq == 0:
                valid_loss, valid_acc = [], []
                valid_cnt = 0
                for fs, ls in self._valid_ds:
                    lo, ma = self._test_step(fs, ls)
                    valid_loss.append(lo)
                    valid_acc.append(ma)
                    valid_cnt += len(fs)
                avg_valid_loss = np.array(valid_loss).sum() / valid_cnt
                avg_valid_acc = np.array(valid_acc).sum() / valid_cnt
                print('Global model ===> VALID avg loss: %.6f, avg acc: %.6f' % (avg_valid_loss, avg_valid_acc))
                self._valid_losses.append(avg_valid_loss)
                self._valid_acc.append(avg_valid_acc)
        
        # 底下的 history 應該不會用到就先沒加

        '''
        try:
            for epoch in range(1 if DEBUG else epochs):
                logging.info(f'Epoch {epoch}/{epochs}')

                sample_count = 0
                total_loss = 0.
                total_match = 0

                for features, labels in self._train_ds:
                    if DEBUG and steps > 300:
                        break

                    loss, match = self._train_step(features, labels)  # batch loss
                    total_loss += loss
                    sample_count += len(features)
                    avg_train_loss = total_loss / sample_count
                    self._train_losses.append(avg_train_loss)

                    total_match += match
                    avg_train_acc = total_match / sample_count
                    self._train_acc.append(avg_train_acc)

                    if log_freq > 0 and steps % log_freq == 0:
                        logging.info('Epoch %d, step %d, avg loss %.6f, avg acc %.6f'
                                     % (epoch, steps, avg_train_loss, avg_train_acc))

                    if valid_freq > 0 and steps % valid_freq == 0:
                        logging.info(f'===> Step: {steps}, evaluating on VALID...')
                        valid_loss, valid_acc = [], []
                        valid_cnt = 0
                        for fs, ls in self._valid_ds:
                            lo, ma = self._test_step(fs, ls)
                            valid_loss.append(lo)
                            valid_acc.append(ma)
                            valid_cnt += len(fs)

                        avg_valid_loss = np.array(valid_loss).sum() / valid_cnt
                        avg_valid_acc = np.array(valid_acc).sum() / valid_cnt
                        logging.info('===> VALID avg loss: %.6f, avg acc: %.6f' % (avg_valid_loss, avg_valid_acc))
                        self._valid_losses.append(avg_valid_loss)
                        self._valid_acc.append(avg_valid_acc)
                    steps += 1
        except Exception as e:
            raise Exception(e)
        finally:
            history = {
                'epoch_steps': steps / epochs,
                'valid_freq': valid_freq,
                'train_loss': self._train_losses,
                'train_acc': self._train_acc,
                'valid_loss': self._valid_losses,
                'valid_acc': self._valid_acc
            }

            with open(history_path, 'wb') as fw:
                pickle.dump(history, fw)
        
        '''

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        tf.saved_model.save(self._model, model_dir)

        logging.info(f'After training {epochs} epochs, '
                     f'save model to {model_dir}, train logs to {history_path}.')

def main(_):
    s = time.time()
    demo = TF(pkt_bytes=256, pkt_num=20, model='pbcnn',
            # demo data
            #   train_path='../data/demo_tfrecord/_train',
            #   valid_path='../data/demo_tfrecord/_valid',
            #   test_path='../data/demo_tfrecord',
            # real data
              train_path='/trainingData/sage/CIC-IDS2018/tfrecord/train',
              valid_path='/trainingData/sage/CIC-IDS2018/tfrecord/valid',
              test_path='/trainingData/sage/CIC-IDS2018/tfrecord/test',
              batch_size=1,
              num_class=15)
    # There are two models can be choose, "pbcnn" and "en_pbcnn".
    demo.init()
    # demo.fit(1)
    # print(demo._predict())
    demo.train(epochs=15)
    logging.info(f'cost: {(time.time() - s) / 60} min')

if __name__ == '__main__':
    app.run(main)

