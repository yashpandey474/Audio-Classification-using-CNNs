import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 43, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.attention = nn.Linear(128 * 16 * 43, 1)  # Attention layer

    def forward(self, x):
        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output for attention mechanism
        x_flatten = x.view(x.size(0), -1)

        # Attention mechanism
        attention_weights = F.softmax(self.attention(x_flatten), dim=1)
        attention_output = torch.mul(x_flatten, attention_weights)

        # Fully connected layers
        x = F.relu(self.fc1(attention_output))
        x = self.fc2(x)
        return x
    
class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate input size for fully connected layers
        self.fc_input_size = self._calculate_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _calculate_fc_input_size(self):
        # Calculate the size of the flattened output after convolution and pooling
        with torch.no_grad():
            x = torch.zeros(1, 3, self.input_shape[0], self.input_shape[1])  # Create dummy input tensor
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = self.pool(nn.functional.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_size)  # Flatten the output for fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SimplifiedResNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimplifiedResNet, self).__init__()
        self.in_channels = 32  # Reduced number of initial channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(ResNetBlockSimple, self.in_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(ResNetBlockSimple, self.layer1[0].conv2.out_channels, blocks=2, stride=2)
        self.layer3 = self._make_layer(ResNetBlockSimple, self.layer2[0].conv2.out_channels, blocks=2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(self.layer3[0].conv2.out_channels, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetBlockSimple(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlockSimple, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    

