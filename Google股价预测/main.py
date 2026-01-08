import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

train_df = pd.read_csv('data/google_train_data.csv')
test_df = pd.read_csv('data/google_test_data.csv')

if 'Volume' in train_df.columns:
    train_df['Volume'] = train_df['Volume'].str.replace(',', '').str.strip().astype(float)
    test_df['Volume'] = test_df['Volume'].str.replace(',', '').str.strip().astype(float)

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)

train_data = train_df[['Close']].values
test_data = test_df[['Close']].values

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


time_step = 60  # 用过去 60 天预测未来
X_train, y_train = create_dataset(train_data_scaled, time_step)
X_test, y_test = create_dataset(test_data_scaled, time_step)

# 转为 PyTorch Tensor
X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (samples, time_step, 1)
y_train = torch.FloatTensor(y_train).unsqueeze(-1)  # (samples, 1)
X_test = torch.FloatTensor(X_test).unsqueeze(-1)
y_test = torch.FloatTensor(y_test).unsqueeze(-1)

print(f"\n训练集:  X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集: X_test={X_test.shape}, y_test={y_test.shape}")


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc1(out)
        out = self.fc2(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 50
train_losses = []
test_losses = []

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

model.eval()
with torch.no_grad():
    train_predict = model(X_train.to(device)).cpu().numpy()
    test_predict = model(X_test.to(device)).cpu().numpy()

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy())

train_rmse = np. sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
test_mae = mean_absolute_error(y_test_actual, test_predict)
test_r2 = r2_score(y_test_actual, test_predict)

print(f"\n========== 评估结果 ==========")
print(f"训练集 RMSE: {train_rmse:.2f}")
print(f"测试集 RMSE: {test_rmse:.2f}")
print(f"测试集 MAE:  {test_mae:.2f}")
print(f"测试集 R²:    {test_r2:.4f}")

plt.figure(figsize=(12, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('训练与测试损失曲线')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

# 训练集
plt.plot(np.arange(len(train_predict)), y_train_actual, label='训练集真实值', color='blue')
plt.plot(np.arange(len(train_predict)), train_predict, label='训练集预测值', color='orange', alpha=0.7)

# 测试集
test_start = len(train_predict)
plt.plot(np.arange(test_start, test_start + len(test_predict)),
         y_test_actual, label='测试集真实值', color='green')
plt.plot(np.arange(test_start, test_start + len(test_predict)),
         test_predict, label='测试集预测值', color='red', alpha=0.7)

plt.axvline(x=test_start, color='black', linestyle='--', label='训练/测试分界')
plt.xlabel('样本索引')
plt.ylabel('股票收盘价')
plt.title('LSTM 股票价格预测结果')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

torch.save(model.state_dict(), 'lstm_stock_model.pth')
print("\n模型已保存为 lstm_stock_model.pth")