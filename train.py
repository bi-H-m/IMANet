import os          # 通过os模块，可以执行文件和目录操作，运行系统命令，管理进程等
import time        # 处理与时间相关的功能
import datetime    # 处理日期和时间
import random      # 随机数生成相关功能的语句
import numpy as np
from glob import glob   # 用于查找文件路径名匹配指定模式的所有文件列表
import albumentations as A   # 用于图像增强的库，提供了丰富的图像增强方法，包括但不限于几何变换、颜色变换、像素级变换等
import cv2     # 提供了丰富的图像处理和计算机视觉功能，包括图像加载、显示、处理、特征检测、目标跟踪等
import torch
# from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader   # 用于加载和处理数据以供训练和测试深度学习模型
from torchvision import transforms   # transforms模块提供了一系列用于图像数据预处理和增强的转换操作

from utils import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, rle_decode, print_and_save, load_data
    )
from model.model import FANet
from loss import DiceBCELoss


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        # cv2.imread函数会将图像文件读取为一个NumPy数组
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        # 利用预先定义的变换操作对图像和掩模进行处理，然后更新image和mask变量以存储处理后的图像和掩模数据
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        # 使用NumPy库中的transpose函数对图像进行维度转换
        image = np.transpose(image, (2, 0, 1))
        # 对图像数据进行归一化处理，将像素值缩放到[0, 1]的范围内
        image = image/255.0
        image = image.astype(np.float32)


        mask = cv2.resize(mask, self.size)
        # axis=0:指定了要在哪个轴上进行维度扩展，这里是在第0个轴上进行扩展，通常表示在数组的最外层添加一个新的维度，即(1,H,W)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, mask, optimizer, loss_fn, device):
    epoch_loss = 0
    return_mask = []

    model.train()
    # i对应迭代次数，x对应输入的图片，y对应标签
    # 循环时，每次迭代会处理一个批次的数据，x包含了这个批次中所有图像的数据，而y包含了这个批次中所有标签的数据
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        b, c, h, w = y.shape
        m = []


        for edata in mask[i*b : i*b+b]:
            # 将循环中获取的每个元素edata转换为字符串，并使用空格将它们连接起来，最终赋值给edata
            edata = " ".join(str(d) for d in edata)
            edata = str(edata)
            edata = rle_decode(edata, size)
            # 使用NumPy中的expand_dims函数将数组edata在指定的axis=0上进行扩展，即在第0维上增加一个维度
            edata = np.expand_dims(edata, axis=0)
            m.append(edata)

        # 变量m最初是一个列表，其中包含多个NumPy数组。将变量m转换为NumPy数组，并指定数据类型为int32
        m = np.array(m, dtype=np.int32)
        m = np.transpose(m, (0, 1, 3, 2))
        # 将NumPy数组m转换为PyTorch Tensor
        m = torch.from_numpy(m)
        m = m.to(device, dtype=torch.float32)

        # 通过optimizer.zero_grad()可以确保在每个mini-batch计算梯度前将之前的梯度清零，从而避免梯度累积的问题，保证每个mini-batch的梯度计算是独立的
        optimizer.zero_grad()
        # 通过模型model对输入数据x和m进行前向传播计算，得到模型的预测输出y_pred
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        # 在前向传播过程中，我们计算了模型的预测输出和实际标签之间的损失值，然后通过调用loss.backward()方法，可以自动计算出各个参数对损失的梯度
        # 一旦梯度计算完成，我们就可以利用这些梯度值来更新模型的参数，使得模型能够朝着减小损失的方向优化
        loss.backward()
        # 根据计算得到的梯度来更新模型参数的方法
        optimizer.step()
        # scheduler.step()  # 根据当前迭代更新学习率

        # 没有记录梯度信息
        with torch.no_grad():
            # 对模型输出进行sigmoid转换，并将结果转换为NumPy数组
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy()

            for py in y_pred:
                # 将数组py中的维度为1的轴（在这里是第0个轴）去除掉，从而降低数组的维度
                # axis=0实际上是告诉函数在数组的第一个维度上进行判断和操作
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                py = rle_encode(py)
                return_mask.append(py)

        epoch_loss += loss.item()

    # 计算每个epoch的平均损失值
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, return_mask


def evaluate(model, loader, mask, loss_fn, device):
    epoch_loss = 0
    return_mask = []

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            b, c, h, w = y.shape
            m = []
            for edata in mask[i*b : i*b+b]:
                edata = " ".join(str(d) for d in edata)
                edata = str(edata)
                edata = rle_decode(edata, size)
                edata = np.expand_dims(edata, axis=0)
                m.append(edata)

            m = np.array(m)
            m = np.transpose(m, (0, 1, 3, 2))
            m = torch.from_numpy(m)
            m = m.to(device, dtype=torch.float32)


            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy()

            for py in y_pred:
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                py = rle_encode(py)
                return_mask.append(py)

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, return_mask


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    # 获取当前的日期和时间，并将其转换为字符串格式存储在datetime_object变量中
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = 8
    num_epochs = 100
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset """
    # path = "../data/isic2018"
    path = r"E:\ymy\data\isic2018"
    # 得到训练集和验证集的图像及掩码的完整文件路径列表
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)


    """ Data augmentation: Transforms """
    '''
    transform = A.Compose([
        # Rotate表示对图像进行旋转操作，limit=35表示旋转的角度限制为正负35度，p=0.3表示以0.3的概率应用这个旋转操作
        A.Rotate(limit=35, p=0.3),
        # 水平翻转
        A.HorizontalFlip(p=0.3),
        # 垂直翻转
        A.VerticalFlip(p=0.3),
        # 对图像进行粗粒度的随机区域遮挡。参数p=0.3表示以30%的概率应用这个遮挡操作，max_holes=10表示最大遮挡区域的数量为10个，max_height=32和max_width=32则表示每个遮挡区域的最大高度和宽度为32个像素
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])
    '''
    transform = A.Compose([
        # 随机裁剪
        A.RandomCrop(width=256, height=256, p=0.3),
        # 水平翻转
        A.HorizontalFlip(p=0.3),
        # 垂直翻转
        A.VerticalFlip(p=0.3),
        # 旋转
        A.Rotate(limit=35, p=0.3),
        # 弹性变换
        A.ElasticTransform(p=0.3, alpha=1, sigma=50, alpha_affine=None),
        # 网格失真
        A.GridDistortion(p=0.3, num_steps=5, distort_limit=0.3),
        # 光学失真
        A.OpticalDistortion(p=0.3, distort_limit=0.3, shift_limit=0.3),
        # 灰度转换
        A.ToGray(p=0.3),
        # 随机亮度和对比度
        A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
        # 随机通道
        A.ChannelShuffle(p=0.3),
        # 粗粒度的随机区域遮挡
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])


    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    # data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    # print_and_save(train_log_path, data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    model = FANet()
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 'min' 表示调度器要监测的指标是最小化的，当这个指标不再减小时，调度器将减小学习率
    # patience=5指定了要等待的epochs数，如果在这个epochs数内指标没有改善，学习率将会被调整
    # verbose=True表示在调整学习率时输出一些信息，以便查看学习率的变化情况
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\nDevice: {device}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """
    # 将变量 best_valid_loss 初始化为正无穷大
    best_valid_loss = float('inf')
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, return_train_mask = train(model, train_loader, train_mask, optimizer, loss_fn, device)
        valid_loss, return_valid_mask = evaluate(model, valid_loader, valid_mask, loss_fn, device)
        # 根据验证集上的损失值valid_loss来更新优化器的学习率
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data_str = f"Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)
            # 当调用model.state_dict()时，PyTorch会返回一个字典对象，其中包含了所有模型层的参数及其对应的数值
            torch.save(model.state_dict(), checkpoint_path)

            train_mask = return_train_mask
            valid_mask = return_valid_mask

        end_time = time.time()
        # 用来计算两个时间戳之间的时间差
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print_and_save(train_log_path, data_str)