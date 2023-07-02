import torch
from torch import nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, input_size, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv1d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=6,  # filter size
                stride=1,  # filter movement/step
                padding=5,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        input_size = (1, 32, int(((input_size[1] - 6 + 5 * 2) / 1 + 1) / 2))
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv1d(32, 64, 6, 1, 5),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # output shape (32, 7, 7)
        )
        input_size = (1, 64, int(((input_size[2] - 6 + 5 * 2) / 1 + 1) / 2))
        #		self,flatten = nn.Linear(np.prod(input_size[1:]), 10)
        self.dense1 = nn.Linear(np.prod(input_size[1:]), 1024)
        #		self,dense1_1 = nn.Linear(1024, 10)
        self.dense2 = nn.Linear(1024, 25)
        self.cnn_out = nn.Linear(25, num_class)  # fully connected layer, output number of classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        mid = self.dense1(x)
        x = self.dense2(mid)
        cnn_output = self.cnn_out(x)
        return cnn_output  # return mid for visualization



