import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 从 CSV 加载数据
processed_data = pl.read_csv('processed.csv')

# 2. 提取特征和目标变量
features = ['rsi', 'macd', 'atr', 'log_return', 'ma20', 'volume_change_rate']
features.extend([f'close_lag_{lag}' for lag in [1, 2, 3, 6, 12]])
target = 'future_return'

# 转换为 NumPy 数组
feature_data = processed_data[features].to_numpy()
target_data = processed_data[target].to_numpy()

# 3. 创建时间序列样本
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    X = [torch.tensor(x, dtype=torch.float32) for x in X]
    y = [torch.tensor(label, dtype=torch.float32).unsqueeze(1) for label in y]
    return torch.stack(X), torch.stack(y)

print("Creating time sequences...")
X_seq, y_seq = create_sequences(feature_data)

# 4. 分割数据集
X_train, X_valid, y_train, y_valid = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# 5. 创建 DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 打印数据形状
print(f"Train data shape: {X_train.shape}")
print(f"Valid data shape: {X_valid.shape}")

class CryptoTransformer(nn.Module):
    def __init__(self, input_dim=6, seq_len=24, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model).to(device))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * seq_len, 1)  # 输出未来收益率

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.to(self.positional_encoding.device)  # 使输入数据与 positional_encoding 在同一设备上
        x = self.embedding(x) + self.positional_encoding
        x = x.permute(1, 0, 2)  # Transformer 需要 (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 恢复为 (batch_size, seq_len, d_model)
        x = x.reshape(x.size(0), -1)  # 展平
        return self.fc(x)


def prepare_transformer_data(data, features, seq_len=24):
    """
    将数据转换为 Transformer 需要的格式
    :param data: 输入数据（Polars DataFrame）
    :param features: 特征列表
    :param seq_len: 时间序列长度
    :return: DataLoader
    """
    # 转换为 NumPy 数组
    feature_data = data[features].to_numpy()
    target_data = data['future_return'].to_numpy()

    # 创建时间序列样本
    X, y = [], []
    for i in range(len(feature_data) - seq_len):
        X.append(feature_data[i:i + seq_len])
        y.append(target_data[i + seq_len])

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 创建 DataLoader
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=False)


def train_transformer(model, train_loader, valid_loader, epochs=50, lr=0.001):
    """
    训练 Transformer 模型
    :param model: Transformer 模型
    :param train_loader: 训练数据 DataLoader
    :param valid_loader: 验证数据 DataLoader
    :param epochs: 训练轮数
    :param lr: 学习率
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in trange(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证集评估
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                valid_loss += criterion(outputs.squeeze(), y_batch).item()

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Valid Loss: {valid_loss / len(valid_loader):.4f}")


# 定义模型
model = CryptoTransformer(input_dim=len(features))
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_transformer(model, train_loader, valid_loader)

torch.save(model,'transformer_model.pt')