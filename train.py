import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from conformer import EEG_Conformer
from densecnn import DenseCNN
from transformer import TransformerClassifier
from lstm import LSTMModel

from pytorchtools import EarlyStopping
import copy
import matplotlib.pyplot as plt
import numpy as np

model_type = "conformer"


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.init()


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

# 定义训练函数
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()  # 将模型设为训练模式
    train_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs)
        loss = criterion(outputs, labels.float())
        train_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
        preds = torch.round(outputs)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_loss /= total
    return train_loss, train_acc

# 定义验证函数
def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # 将模型设为评估模式
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item() * inputs.size(0)
            preds = torch.round(outputs)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_loss /= total
    return val_loss, val_acc

# 设置参数
batch_size = 128
learning_rate = 0.00001
num_epochs = 500

# 准备数据
print("1. Prepare dataset...")
train_set = torch.load("/content/drive/MyDrive/NTU-research/Comformer/EEG_DATA/Epilept/train.pt")
train_data = train_set["samples"].float()
train_labels = train_set["labels"]
train_dataset = MyDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_set = torch.load("/content/drive/MyDrive/NTU-research/Comformer/EEG_DATA/Epilept/val.pt")
val_data = val_set["samples"].float()
val_labels = val_set["labels"]
val_dataset = MyDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 创建模型和优化器
print("2. Inital model...")
if model_type == "conformer":
    model = EEG_Conformer()
# elif model_type == "dcrnn":
#     model = DCRNNModel_classification()
elif model_type == "densecnn":
    model = DenseCNN()
elif model_type == "lstm":
    model = LSTMModel()
# elif model_type == "cnnlstm":
#     model = CNN_LSTM(2)
elif model_type == "transformer":
    model = TransformerClassifier()
print("Using model:", model_type)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# 将模型和数据移到GPU上
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
model.to(device)

print("3. Start training...")
best_acc=0.0
valid_acc = []  # 定义一个空列表用于存储验证准确率

# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []


# 复制模型的参数
best_model_wts=copy.deepcopy(model.state_dict())

# early_stopping的初始化
early_stopping = EarlyStopping(patience=12, verbose=True)

# learning rate decay的初始化
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-8)

print(f"The batch size is: {batch_size}")
# 开始迭代
for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    # learning rate decay
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    # calculate average loss over an epoch
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(val_loss)


    # 将当前的验证准确率添加到列表中
    valid_acc.append(val_acc)

    # 拷贝模型最高精度下的参数
    if valid_acc[-1]>best_acc:
        best_acc=valid_acc[-1]
        best_model_wts=copy.deepcopy(model.state_dict())
        print(f"Now the best model epoch is {epoch+1}, and best val_acc is {best_acc:.4f}")
        torch.save(best_model_wts, 'model104.pth')


    # patience次都不下降则停止训练
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("The Early stopping epoch is this：", epoch)
        break



# Visualization

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')