#第一次作业
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
from model1 import *
#准备数据集
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#数据增强处理，适用于数据量比较少的实验
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
#对测试集不需要数据增强处理
transform1 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_data = torchvision.datasets.CIFAR10("./run_data/datasets", train=True, transform=transform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./run_data/datasets", train=False, transform=transform1,
                                         download=True)
#数据集长度
length_train_data = len(train_data)
length_test_data = len(test_data)
print("训练数据集的长度为：{}".format(length_train_data))
print("测试数据集的长度为:{}".format(length_test_data))

#利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=100)

#创建网络模型
network = Network()
if torch.cuda.is_available():
    network = network.cuda()
#损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
#优化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
#设置训练网络的一些参数

#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 100

#添加tensorboard
writer = SummaryWriter("../logs/logs_train1")
start_time = time.time()
for i in range(epoch):
    print("--------第{}轮训练开始---------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = network(imgs)
        loss = loss_fn(outputs, targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数:{}, loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

   #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = network(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的accuracy:{}".format(total_accuracy/length_test_data))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/length_test_data, total_test_step)
    total_test_step += 1
    #将每次训练的模型保存下来
    torch.save(network, "../network/network_{}.pth".format(i))

    print("模型已保存")

writer.close()

