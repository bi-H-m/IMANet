import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Load the Kvasir-SEG dataset """
'''
    def generate_file_list(path, train_ratio=0.88):
        image_names = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "images")) if f.endswith('.jpg')]
        random.shuffle(image_names)

        split_idx = int(len(image_names) * train_ratio)
        train_names = image_names[:split_idx]
        valid_names = image_names[split_idx:]

        with open(os.path.join(path, 'train.txt'), 'w') as f:
            for name in train_names:
                f.write(name + '\n')

        with open(os.path.join(path, 'val.txt'), 'w') as f:
            for name in valid_names:
                f.write(name + '\n')

    # 生成训练集和验证集文件列表
    generate_file_list(path)

    def load_data(path):
    def load_names(path, file_path):
        f = open(file_path, "r")
        # 通过[:-1]的切片操作去掉了列表中最后一个空字符串元素，这通常是因为在文件的末尾有一个额外的换行符导致的
        data = f.read().split("\n")[:-1]
        images = [os.path.join(path, "images", name) + ".jpg" for name in data]
        masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]
        return images, masks

    train_names_path = f"{path}/train"
    valid_names_path = f"{path}/val"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)
'''

def load_data(path):
    def load_images_masks(subset_path):
        images_dir = os.path.join(subset_path, 'images')
        masks_dir = os.path.join(subset_path, 'masks')

        # 获取文件夹下所有.jpg文件的完整路径
        images = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]
        masks = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.png')]

        # 确保images和masks的文件数量一致，按名称排序以保证一一对应
        images.sort()
        masks.sort()

        return images, masks

    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'val')

    train_x, train_y = load_images_masks(train_path)
    valid_x, valid_y = load_images_masks(valid_path)

    return (train_x, train_y), (valid_x, valid_y)


""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


'''
def rle_encode(x):

    mpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list


    # dots变量中存储了二维数组x中值为1的元素在展平后的一维数组中的索引位置，即转置，展平，找为1的索引元组
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
'''


# 经过RLE编码，得到的结果为列表，表示前景（1）开始的位置及长度
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as list format
    '''
    run_lengths = []
    pixels = im.flatten(order='F')
    # 在NumPy数组pixels的两侧各添加一个值为0的元素，然后将它们连接到数组的两侧
    pixels = np.concatenate([[0], pixels, [0]])
    # 找出数组 pixels 中不同值的起始索引位置
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs数组中存储了相邻不同值之间的长度
    runs[1::2] -= runs[::2]
    run_lengths = runs.tolist()
    return run_lengths


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''

    # 按空格进行分割，分别是起始位置和对应的长度
    s = mask_rle.split()
    # 将RLE编码后的列表s中的奇数索引和偶数索引分别提取出来，并分别存储为starts和lengths两个numpy数组
    # 奇数索引对应的是起始位置信息，偶数索引对应的是长度信息
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    # 通过将起始位置starts和长度lengths相加来得到每个区间的结束位置信息，并将结果存储在ends变量中
    ends = starts + lengths
    # 创建了一个长度为 shape[0] * shape[1] 的一维数组，所有元素的初始数值都是 0
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # zip(starts, ends) 用于同时迭代starts和ends两个数组中对应位置的元素
    # 结束位置的索引是作为切片右边界使用的值，不包含在切片结果中
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
    # return img.reshape(shape, order='F')


""" Initial mask build using Otsu thresholding. """
def init_mask(images, size):
    def otsu_mask(image, size):
        # 使用cv2.imread()函数加载这张图片。第一个参数是图片的文件路径，第二个参数cv2.IMREAD_GRAYSCALE指定将图片以灰度形式加载
        # img将是一个代表灰度图像的NumPy数组
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        # 使用了OpenCV库中的GaussianBlur()函数来对图像进行高斯模糊处理
        # (5, 5)指定了高斯核的大小，这里的值表示高斯核的宽和高都是5。最后一个参数0表示在x和y方向上的标准差，如果为0，则由函数根据核函数的尺寸自动计算
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        '''
        利用OpenCV库中的threshold()函数对经过高斯模糊处理后的图像进行阈值化操作，并使用OTSU算法自动选择最佳阈值
        # 0表示手动设置的阈值，由于使用了cv2.THRESH_OTSU，这里的阈值值将被忽略
        # 255表示最大灰度值，即在阈值化后，小于阈值的像素值会被设为0，大于阈值的像素值会被设为255
        # cv2.THRESH_BINARY+cv2.THRESH_OTSU指定了使用OTSU算法确定阈值，并将图像二值化
        ret将返回由OTSU算法选择的最佳阈值，th将包含阈值化后的二值图像数据，其中像素值只有0和255两种
        '''
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # astype()函数用于实现数组类型的转换
        th = th.astype(np.int32)
        # 将二值图像数据th进行了归一化处理，将所有的像素值都除以255.0，使得像素值范围在0到1之间
        th = th / 255.0
        # 将二值图像数据th中的像素值与0.5进行比较，生成一个布尔类型的numpy数组，其中True表示像素值大于0.5，False表示像素值小于或等于0.5
        th = th > 0.5
        # 将布尔值True和False转换为整数1和0
        th = th.astype(np.int32)
        return img, th

    mask = []
    for image in tqdm(images, total=len(images)):
        # 将完整的路径列表按"/"进行划分，截取最后一部分内容，例如1.jpg
        name = image.split("/")[-1]
        # i表示灰度图，而m表示阈值处理后的二值化图
        i, m = otsu_mask(image, size)
        # cv2.imwrite(f"mask/{name}", np.concatenate([i, m*255], axis=1))
        m = rle_encode(m)
        mask.append(m)

    return mask
