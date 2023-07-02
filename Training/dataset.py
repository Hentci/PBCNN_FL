import glob

from torch.utils.data import Dataset
import numpy as np


def load_processed_data(mode, img_shape, over, under, node_id, num_node, alpha):
    path = f'/home/christin/Workplace/traffic_datasets/Sampling/CSE-CIC-IDS-2018/processed/' \
           f'{img_shape[0]}_{img_shape[1]}_flows'

    if mode == 'train_resample':
        data_file = path + f'/resampled_train/all_{over}_{under}.npy'
        label_file = path + f'/resampled_train/all_{over}_{under}_label.npy'
        data = np.load(data_file)
        data = np.reshape(data, (len(data), 1, -1))
        label = np.load(label_file)
    elif mode == 'train':
        files = glob.glob(f'{path}/train/*.npy')
        data = []
        label = []
        for f in sorted(files):
            if 'label' in f:
                tmp_label = np.load(f)
                label.extend(tmp_label)
            else:
                tmp_data = np.load(f)
                tmp_data = np.reshape(tmp_data, (len(tmp_data), 1, -1))
                data.extend(tmp_data)
    else:
        files = glob.glob(f'{path}/test/*.npy')
        data = []
        label = []
        for f in sorted(files):
            if 'label' in f:
                tmp_label = np.load(f)
                label.extend(tmp_label)
            else:
                tmp_data = np.load(f)
                tmp_data = np.reshape(tmp_data, (len(tmp_data), 1, -1))
                data.extend(tmp_data)

    data = np.asarray(data)
    label = np.asarray(label)

    return data, label


class IDSDataset(Dataset):
    def __init__(self, mode, img_shape, over=0, under=0, node_id='all', num_node=0, alpha=-1):
        # load data
        self.data, self.label = load_processed_data(mode, img_shape, over, under, node_id, num_node, alpha)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


