import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 數據預處理
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),# 對圖片進行隨機裁減並縮放至指定大小
    transforms.RandomRotation(20), # 隨機旋轉角度
    transforms.RandomHorizontalFlip(p=0.5), # 隨機水平翻轉
    transforms.ToTensor()
])

# 讀取數據
root = 'dataset'
train_dataset = datasets.ImageFolder(root + '/train', transform)
test_dataset = datasets.ImageFolder(root + '/test', transform)

# 導入數據
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

# 從導入的資料得出分類名稱
classes = train_dataset.classes
classes_index = train_dataset.class_to_idx
print(classes)
print(classes_index)

model = models.resnet18(pretrained = True)
print(model)

model.fc = torch.nn.Linear(512, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 設定學習率
LR = 0.0001
# 定義損失函數
entropy_loss = nn.CrossEntropyLoss()
# 定義優化器
optimizer = optim.SGD(model.parameters(), LR, momentum=0.9)

def train():
    # 設定為訓練狀態
    model.train()
    for i, data in enumerate(train_loader):
        # 獲取數據與對應的標籤
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 獲得模型預測結果
        out = model(inputs)
        # 透過損失函數計算誤差
        loss = entropy_loss(out, labels)
        # 將梯度歸零
        optimizer.zero_grad()
        # 計算梯度
        loss.backward()
        # 透過優化器修改權重值
        optimizer.step()


def test():
    # 設定為評估狀態
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        # 獲取數據與對應的標籤
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 獲得模型預測結果
        out = model(inputs)
        # 獲得最大值，以及最大值所在的位置
        _, predicted = torch.max(out, 1)
        # 計算正確預測的數量
        correct += (predicted == labels).sum()
    print("Test acc: {0}".format(correct.item() / len(test_dataset)))

    correct = 0
    for i, data in enumerate(train_loader):
        # 獲取數據與對應的標籤
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 獲得模型預測結果
        out = model(inputs)
        # 獲得最大值，以及最大值所在的位置
        _, predicted = torch.max(out, 1)
        # 計算正確預測的數量
        correct += (predicted == labels).sum()
    print("Train acc: {0}".format(correct.item() / len(train_dataset)))

for epoch in range(0, 10):
    print('epoch:',epoch)
    train()
    test()

torch.save(model.state_dict(), 'final_model.pth')
