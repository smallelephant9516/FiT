import torch
import torch.nn as nn

class ThreeLayerCNN1D(nn.Module):
    def __init__(self, d_channels):
        super(ThreeLayerCNN1D, self).__init__()

        # Define the layers
        # Using padding=1 with kernel_size=3 to ensure spatial size remains the same
        self.conv1 = nn.Conv1d(in_channels=d_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=d_channels, kernel_size=3, stride=1, padding=1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class SingleLayerCNN1D(nn.Module):
    def __init__(self, d_channels, kernel_size=3):
        super(SingleLayerCNN1D, self).__init__()

        # Single convolutional layer
        # Using padding=1 with kernel_size=3 to ensure spatial size remains the same
        self.conv1 = nn.Conv1d(in_channels=d_channels, out_channels=d_channels, kernel_size=kernel_size, stride=1,
                               padding=1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return x


class PositionalEmbeddingGenerator1D(nn.Module):
    def __init__(self, in_channels, av_width, layers, dims, learned_weight=True):
        super(PositionalEmbeddingGenerator1D, self).__init__()
        self.learned_weight = learned_weight
        if learned_weight is True:
            self.generator = SingleLayerCNN1D(d_channels=in_channels, kernel_size=av_width)

    def forward(self, data):
        b, length, in_channles = data.shape
        data_cnn_input = data.permute(0, 2, 1)
        positional_emb = self.generator(data_cnn_input)
        positional_emb = positional_emb.permute(0, 2, 1)
        return positional_emb