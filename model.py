import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # 输入通道=1
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)  # 输入通道=32
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 5, 100)
        self.fc2 = nn.Linear(100, 10)
 
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 5)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor
def get_model(input_channels=1, num_classes=10):
    """
    创建CNN模型的工厂函数
    
    Args:
        input_channels: 输入图像的通道数，默认为1（灰度图像）
        num_classes: 分类类别数，默认为10
        
    Returns:
        CNN模型实例
    """
    model = Mnist_CNN()
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params}")
    
    return model