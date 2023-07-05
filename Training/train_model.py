import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import ConfusionMatrix
from tqdm import tqdm

from dataset import IDSDataset
from models import CNN

NUM_NODE = 16

NUM_CLASS = 2
DEVICE = 'cuda:0'
EPOCH = 200
O = 1
U = 2000


def train(node_id='all', alpha=0.1, batch_size=32, img_shape=(60, 3)):

    # init log
    # path = f'Log/all/{O}_{U}'
    path = f'Log/local/{O}_{U}/img_shape/{img_shape[0]}_{img_shape[1]}'
    if not os.path.exists(path):
        os.makedirs(path)
    class_file = f'{path}/class_validation_log_{node_id}.csv'
    if os.path.exists(class_file):
        os.remove(class_file)
    with open(class_file, 'a') as f:
        f.write(f'class acc, precision, recall, f1-score, acc\n')

    # dataset and dataloader
    training_data = IDSDataset(mode='train', img_shape=img_shape, over=O, under=U, node_id=node_id,
                               num_node=NUM_NODE, alpha=alpha)
    train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    test_data = IDSDataset(mode='test', img_shape=img_shape, over=O, under=U)
    test_dataloader = DataLoader(dataset=test_data, batch_size=4096, shuffle=False)

    # load model
    pshape = [1, img_shape[0] * img_shape[1]]
    model = CNN([pshape[0], pshape[1]], NUM_CLASS)
    model.to(DEVICE)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    for e in range(EPOCH):
        print(f'Epoch: {e + 1}')
        with open(class_file, 'a') as f:
            f.write(f'Epoch {e+1}\n')

        # training
        train_loss = 0
        total = 0
        correct = 0
        with tqdm(train_dataloader) as t:
            for i, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.float().to(DEVICE), labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # loss
                train_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                predicted = outputs.argmax(1)
                correct += predicted.eq(labels).sum().item()

                # log
                acc = 100. * correct / total
                t.set_description(
                    f'average_training_loss on node {node_id}: {train_loss / total:.4f}, accuracy: {acc:.4f}')

        # testing
        test_loss = 0
        total = 0
        correct = 0
        class_correct = [0] * NUM_CLASS
        class_predict = [[0 for _ in range(NUM_CLASS)] for _ in range(NUM_CLASS)]
        class_size = [0] * NUM_CLASS
        confmat = torch.zeros(NUM_CLASS, NUM_CLASS).to(DEVICE)
        with torch.no_grad() and tqdm(test_dataloader) as t:
            for i, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.float().to(DEVICE), labels.to(DEVICE)

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # test loss
                test_loss += loss.item() * labels.size(0)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # compute correct num for each class
                for j in range(NUM_CLASS):
                    class_correct[j] += ((predicted == labels) * (labels == j)).sum()
                    class_size[j] += (labels == j).sum()

                # compute confusion matrix and other metrics
                metric = ConfusionMatrix(task="multiclass", num_classes=NUM_CLASS).to(DEVICE)
                tmp_confmat = metric(predicted, labels)
                confmat = confmat + tmp_confmat

                # log
                acc = 100. * correct / total
                t.set_description(f'average_test_loss on node {node_id}: {test_loss / total:.4f}, accuracy: {acc:.4f}')


        # print class accuracy
        print('*' * 50)
        # new version compute metric for per class
        tp = torch.zeros(NUM_CLASS)
        tn = torch.zeros(NUM_CLASS)
        fp = torch.zeros(NUM_CLASS)
        fn = torch.zeros(NUM_CLASS)
        mc_acc = torch.zeros(NUM_CLASS)
        mc_precision = torch.zeros(NUM_CLASS)
        mc_recall = torch.zeros(NUM_CLASS)
        mc_f1 = torch.zeros(NUM_CLASS)
        for j in range(NUM_CLASS):
            tp[j] = confmat[j, j]
            fp[j] = confmat[:, j].sum() - tp[j]
            fn[j] = confmat[j].sum() - tp[j]
            tn[j] = confmat.sum() - tp[j] - fp[j] - fn[j]
            mc_acc[j] = (tp[j] + tn[j]) / (tp[j] + tn[j] + fp[j] + fn[j]) * 100.
            mc_precision[j] = tp[j] / (tp[j] + fp[j]) * 100.
            mc_recall[j] = tp[j] / (tp[j] + fn[j]) * 100.
            mc_f1[j] = (2 * mc_recall[j] * mc_precision[j]) / (mc_recall[j] + mc_precision[j])

            print(f'class {j} acc: {mc_acc[j]:.4f}, precision: {mc_precision[j]:.4f}, recall: {mc_recall[j]:.4f},'
                  f' f1-score: {mc_f1[j]:.4f}')

        avg_mc_acc = torch.mean(mc_acc)
        avg_mc_precision = torch.mean(mc_precision)
        avg_mc_recall = torch.mean(mc_recall)
        avg_mc_f1 = torch.mean(mc_f1)
        print(f'macro-average acc: {avg_mc_acc:.4f}, precision: {avg_mc_precision:.4f}, recall: {avg_mc_recall:.4f},'
              f' f1-score: {avg_mc_f1:.4f}')
        print('*' * 50)

        # save model
        # path = f'weight/local/{O}_{U}/alpha/{alpha}'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # file = f'{path}/model_state_dict_{node_id}.pt'
        # torch.save(model.state_dict(), file, _use_new_zipfile_serialization=False)

        # log
        with open(class_file, 'a') as f:
            for j in range(NUM_CLASS):
                f.write(f'{mc_acc[j]:.4f}, {mc_precision[j]:.4f}, {mc_recall[j]:.4f}, {mc_f1[j]:.4f}\n')
            f.write(f'{avg_mc_acc:.4f}, {avg_mc_precision:.4f}, {avg_mc_recall:.4f}, {avg_mc_f1:.4f}, {acc:.4f}\n')


if __name__ == '__main__':
    train(img_shape=(60, 3))
