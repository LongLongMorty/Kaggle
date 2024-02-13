import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast
# 读取CSV文件
df = pd.read_csv("data/train.csv")

heartbeat_signals = df['heartbeat_signals'].apply(lambda x: [float(i) for i in x.split(',')])
labels = df['label'].values
X = np.array(heartbeat_signals.tolist())
y = labels
# 检查CUDA是否可用，并设置默认设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


batch_size = 256
# 首先，分割出训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 接着，从训练集中分割出验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# 使用SMOTE进行过采样处理类别不平衡，只对训练集进行处理
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 转换为PyTorch张量并移至设备
X_train_tensor = torch.tensor(X_resampled, dtype=torch.float16).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_resampled, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float16).unsqueeze(1).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float16).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 使用DataLoader来批量处理数据
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def calculate_abs_sum(y_true, y_pred):

    abs_sum = np.sum(np.abs(y_pred - y_true), axis=1).mean()
    return abs_sum



class DeeperCNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(DeeperCNNModel, self).__init__()
        # 初始卷积层与之前相同
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # 添加额外的卷积层
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.5)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 调整全连接层以匹配最后一个卷积层的输出
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 初始化模型、损失函数、优化器和混合精度缩放器
num_epochs = 30
model = DeeperCNNModel(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        # 自动混合精度
        with autocast():
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        # 梯度缩放和反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
    model.eval()  # 设置模型为评估模式
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X_val, y_val in val_loader:  # 确保你有一个验证集加载器
            X_val = X_val.to(device, dtype=torch.float32)
            y_val = y_val.to(device)

            outputs = model(X_val)
            probabilities = torch.softmax(outputs, dim=1)
            y_pred = probabilities.cpu().numpy()
            y_pred_list.extend(y_pred)

            # 对于独热编码的转换，确保y_val是类别索引形式
            y_val = y_val.cpu().numpy()
            y_true_list.extend(y_val)

    num_classes = 4
    # 计算abs-sum
    y_true_onehot = np.zeros((len(y_true_list), num_classes))
    y_true_onehot[np.arange(len(y_true_list)), y_true_list] = 1
    abs_sum_value = calculate_abs_sum(y_true_onehot, np.array(y_pred_list))
    print(f'Epoch {epoch + 1}, Abs-Sum: {abs_sum_value:.4f}')

if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), 'best_model_state_dict.pth')
        print(f"Saved Best Model at Epoch {epoch+1} with Loss {best_val_loss:.4f}")