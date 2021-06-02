import torch
from torchvision import transforms  # 图像处理
from torch import nn
from torch import optim  # 优化函数
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # 图像读取
from model import my_alexnet
from model import my_vgg16
import random
import matplotlib.pyplot as plt

random.seed(0)

# 若有GPU则使用GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transforms 图像预处理
TRANSFORMS = transforms.Compose(
     [
     transforms.Grayscale(num_output_channels=1), # 转成灰度图像
     transforms.CenterCrop(700),  # 中心裁剪（数据集周围空白太多）
     transforms.Resize((224,224)),  # 图像压缩
     transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])
     ])

# 数据读取
data_dir = r'digit'
data =ImageFolder(data_dir, transform=TRANSFORMS)

# 按照9:1的比例划分训练集和测试集
num_samples = len(data)
num_test = int(0.2 * num_samples)
num_train = len(data) - num_test
train_data, test_data = torch.utils.data.random_split(data, [num_train, num_test])
# print(len(test_data))



# 超参数
NUM_EPOCHS = 40
Learning_rate = 0.001

# 训练
if __name__ == '__main__':

     train_DataLoader = DataLoader(train_data, batch_size=1, shuffle=True)
     test_DataLoader = DataLoader(test_data, batch_size=1, shuffle=False)

     net = my_alexnet()  # 此处使用基于AlexNet设计的CNN，若想使用VGG，请使用下一行的注释代码
     # net = my_vgg16()

     net.to(DEVICE)
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.SGD(net.parameters(), lr=Learning_rate, momentum=0.9)

     train_losses = []  # 训练损失
     train_acces = []  # 训练正确率
     test_losses = []  # 测试损失
     test_acces = []  # 测试正确率

     for epoch in range(NUM_EPOCHS):
          train_loss = 0
          train_acc = 0
          test_loss = 0
          test_acc = 0

          for img, label in train_DataLoader:
               optimizer.zero_grad()

               img = img.to(DEVICE)
               label = label.to(DEVICE)
               net.to(DEVICE)

               out = net(img)  # 前向传播预测值

               loss = criterion(out, label)
               print("train loss:", loss)  # 每个图像都打印一次loss
               loss.backward()
               optimizer.step()

               train_loss += loss.item()

               num_correct = (torch.argmax(out) == label).sum().item()  # 是否正确
               train_acc += num_correct

          for img, label in test_DataLoader:
               net.to(DEVICE)
               img = img.to(DEVICE)
               label = label.to(DEVICE)
               pred = net(img)

               eval_loss = criterion(pred, label)
               print("test loss:", eval_loss)  # 每个图像的测试损失

               test_loss += eval_loss.sum().item()
               num_correct = (torch.argmax(pred) == label).sum().item()  # 是否正确
               test_acc += num_correct

          train_losses.append(train_loss / len(train_DataLoader))
          train_acces.append(train_acc / len(train_DataLoader))
          test_losses.append(test_loss / len(test_DataLoader))
          test_acces.append(test_acc / len(test_DataLoader))

          print('Epoch {} \nTrain Loss {} Train  Accuracy {} \nTest Loss {} Test Accuracy {}'.format(epoch + 1, train_loss / len(train_DataLoader),
                                                                    train_acc / len(train_DataLoader),test_loss / len(test_DataLoader),test_acc / len(test_DataLoader) ))




