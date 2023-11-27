### kaggle手写数字识别pytorch版本实现

#### 1.导入对应的库

```python
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
%matplotlib inline
```

#### 2.数据处理

##### 2.1 将数据转换为tensor，符合pytorch要求

```python
# 读取数据
df = pd.read_csv('data/train.csv')

# 从数据中将标签取出，然后将数据类型转为tensor
labels = torch.from_numpy(df['label'].values)

# 将标签从数据集中删除，得到训练集的输入数据
df.drop('label', axis=1, inplace=True)

# 将训练集转为tensor
imgs = torch.from_numpy(df.values).to(torch.float32)
```

##### 2.2 数据增强

```python
# 数据增强的转换
transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),# 随机缩放后裁剪为28x28
    transforms.ToTensor(), # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 归一化
])

```

##### 2.3 自定义数据集

```python
# 自定义数据集
class CustomDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].view(28, 28, 1).numpy() # 重塑图像
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

##### 2.3 数据加载

```python
# 创建数据加载器
train_dataset = CustomDataset(imgs, labels, transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

```



#### 3. 模型定义

```python
# 模型定义简单的全连接神经网络模型，包含三个隐藏层和一个输出层。
class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平图像
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

```

#### 4.定义损失函数和优化器

```python
# 检测是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化网络并移到设备上
model = Net(784, 300, 100, 10).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### 5.训练网络

```python
# 训练网络
num_epochs = 30
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_acc += (predicted == labels).sum().item() / labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
```

**Loss: 0.0505 ，Accuracy ：0.9839**

**Kaggle得到的成绩为：Top 46%  Score: 0.98682**

#### 6.改进网络结构与训练策略

##### 6.1网络结构改进

```python
 # CNN模型定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

```

**该网络结构是在CNN的基础上加上了dropout操作，防止过拟合操作。**

##### 6.2 数据增强

```python
# 数据增强
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomResizedCrop(28, scale=(0.8, 1.1)),  # 随机缩放
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])
```

