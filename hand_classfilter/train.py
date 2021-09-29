import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
from tensorboardX import SummaryWriter
from utils.logger import Logger
import sys
from utils.pytorchtools import EarlyStopping
import numpy as np
from efficientnet_pytorch import EfficientNet

# 保存日志
sys.stdout = Logger("log/train_log.txt")
# 设置模型训练参数
batch_size = 16
learning_rate = 0.001
epochs = 100
n_classes = 6
# b0:224 b1:240 b2:260 b3:300 b4:380 b5:456 b6:528 b7:600
input_size = 224

data_transform = {
    "train": transforms.Compose([
        transforms.CustomResize((input_size, input_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]),
        # transforms.RandomApply([transforms.RandomRotation(10)]),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),

    "val": transforms.Compose([
        transforms.CustomResize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])}


def create_datasets(data_type: str):

    """
    创建数据集
    """
    # 定义训练集
    data_set = dsets.ImageFolder(root=f'./dataset/hands/{data_type}', transform=data_transform[f'{data_type}'])
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=True,
                                              num_workers=2)
    data_set_number = len(data_set)
    print(f"number of {data_type} sets : {data_set_number}")

    return data_loader, data_set_number


def train_model(model, patience):
    """
    训练模型，每一轮训练后，都进行模型验证，每次保存的结果，都是迄今为止valid loss 最低时候的训练结果
    """
    # 在模型训练时跟踪训练损失
    train_losses = []
    # 在模型训练时跟踪验证损失
    valid_losses = []

    # 初始化early_stopping对象
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # 获取训练集和验证集
    train_loader, train_number = create_datasets('train')
    valid_loader, valid_number = create_datasets('val')

    # 定义损失函数和优化器
    loss_function = tnn.CrossEntropyLoss()
    # loss_function = LabelSmoothSoftmaxCE(label_smooth=0.05, class_num=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 创建可视化对象
    writer = SummaryWriter("./log")

    torch.cuda.synchronize(device)
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_correct = 0
        valid_correct = 0
        # 训练模型
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            # 计算损失

            loss = loss_function(outputs, labels)
            # 统计训练集正确结果
            _, predicted = torch.max(outputs.detach(), dim=1)
            train_correct += torch.eq(predicted, labels).sum().item()
            #
            # print("[E: %d] loss: %f" % (epoch, loss.item()))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证模型
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # 计算损失
                loss = loss_function(outputs, labels)
                # 统计验证集正确结果
                _, predicted = torch.max(outputs.detach(), dim=1)
                valid_correct += torch.eq(predicted, labels).sum().item()
                # 记录验证损失
                valid_losses.append(loss.item())

        # 计算一个epoch后的平均损失
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        # 计算一个epoch后的准确率
        train_accuracy = train_correct / train_number
        valid_accuracy = valid_correct / valid_number
        # 打印loss结果
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} train_accuracy: {train_accuracy: .6f} ' +
                     f'valid_loss: {valid_loss:.6f} valid_accuracy: {valid_accuracy: .6f}')
        print(print_msg)
        # 绘制train_loss和valid_loss曲线图
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
        # 绘制train_accuracy和valid_accuracy曲线图
        writer.add_scalar('train_accuracy', train_accuracy, global_step=epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, global_step=epoch)
        # 清除list以追踪下一个epoch
        train_losses = []
        valid_losses = []

        # early_stopping 传入当前验证集错误率，判断是否提前终止训练
        early_stopping((1 - valid_accuracy), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if epoch == epochs:
            torch.save(model.state_dict(), 'last.pth')
    torch.cuda.synchronize(device)
    end_time = time.time()
    print('time-consuming : %d' % (end_time - start_time))


if __name__ == "__main__":
    model = EfficientNet.from_name('efficientnet-b0')
    t = torch.load("./weights/adv-efficientnet-b0-b64d5a18.pth")
    model.load_state_dict(t)
    feature = model._fc.in_features
    model._fc = tnn.Linear(in_features=feature, out_features=n_classes, bias=True)
    train_model(model=model, patience=20)
