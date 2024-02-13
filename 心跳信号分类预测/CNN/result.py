import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn

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


# 加载模型，确保模型结构与之前保存的一致
# 检查CUDA是否可用，并设置默认设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model = DeeperCNNModel(num_classes=4)
model.load_state_dict(torch.load('best_model_state_dict.pth'))
model.to(device)
model.eval()

# 加载新的数据文件
new_data_df = pd.read_csv("data/testA.csv")
new_heartbeat_signals = new_data_df['heartbeat_signals'].apply(lambda x: np.array([float(i) for i in x.split(',')]))

# 转换为PyTorch张量
X_new = torch.tensor(np.array(new_heartbeat_signals.tolist()), dtype=torch.float32).unsqueeze(1).to(device)

# 数据加载器
batch_size = 128
new_dataset = TensorDataset(X_new)
new_loader = DataLoader(dataset=new_dataset, batch_size=batch_size, shuffle=False)

# 模型推断
predictions = []
with torch.no_grad():
    for X_batch in new_loader:
        X_batch = X_batch[0].to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        predictions.append(probs.cpu().numpy())

predictions = np.vstack(predictions)  # 合并所有批次的预测结果

# 生成输出DataFrame
output_df = pd.DataFrame(predictions, columns=['label_0', 'label_1', 'label_2', 'label_3'])
output_df.insert(0, 'id', new_data_df['id'])

# 保存输出文件
output_df.to_csv("output_predictions.csv", index=False)
