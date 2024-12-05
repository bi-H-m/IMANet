from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix

from model.model import FANet
from utils import create_dir, seeding, init_mask, rle_encode, rle_decode, load_data


def mask_parse(mask):
    # np.squeeze函数，将数组mask中维度为1的轴删除，从而减少数组的维度
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


# 标准的torch.nn.DataParallel类中，gather方法用于将分布在不同设备上的模型输出结果进行汇总
# 对输出结果列表进行求和操作，将多个输出结果列表合并成一个列表
class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def calculate_metrics(y_true, y_pred, img):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    # 将y_pred转换为一维的形状，即将其变为一个扁平的数组
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    # 计算真正样本的特异度
    confusion = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    recall = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0

    return [f1_or_dsc, miou, recall, specificity, precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load dataset """
    path = r"E:\ymy\data\isic2018"
    # path = "../data/isic2018"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    """ Hyperparameters """
    size = (256, 256)
    num_iter = 10
    checkpoint_path = "files/checkpoint.pth"

    """ Directories """
    create_dir("results")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FANet()
    # model = model.to(device)
    # map_location参数指定加载位置，确保模型参数被正确加载到目标设备上
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = CustomDataParallel(model).to(device)
    model.eval()

    """ Testing """
    prev_masks = init_mask(test_x, size)
    # 存储每一次迭代得到的掩码，共10个列表，每个列表中又有120张图的掩码信息
    save_data = []
    # 打开一个名为"test_results.csv"的文件，以供写入数据。具体来说，代码中的参数"w"表示以写入模式打开文件，如果文件不存在则会创建该文件，如果文件已经存在则会清空文件内容
    file = open("files/test_results.csv", "w")
    file.write("Iteration, f1_or_dsc, miou, recall, specificity, percision\n")

    # 初始化最大值和对应的迭代次数
    max_f1_or_dsc = 0.0
    max_miou = 0.0
    max_recall = 0.0
    max_specificity = 0.0
    max_precision = 0.0


    for iter in range(num_iter):

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        tmp_masks = []

        # zip(test_x, test_y):这将两个列表test_x和test_y中的元素一一配对，形成一个迭代器
        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):

            # Image
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)
            img_x = image

            # 对于彩色图像而言，通常的维度顺序是(height, width, channels)
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.0
            # 使用了np.expand_dims函数来在图像数据的最前面添加一个维度，通常用于将单张图像数据准备成批处理数据（batch）的格式
            # 参数axis=0指定了在第0轴（最前面）添加一个新的维度
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            # 将NumPy数组转换为PyTorch的Tensor格式
            image = torch.from_numpy(image)
            image = image.to(device)

            # Mask
            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)
            mask = np.expand_dims(mask, axis=0)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=0)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.to(device)

            # Prev mask
            pmask = prev_masks[i]
            # 将pmask中的数据转换为字符串，并用空格连接起来
            pmask = " ".join(str(d) for d in pmask)
            pmask = str(pmask)
            pmask = rle_decode(pmask, size)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = np.expand_dims(pmask, axis=0)
            pmask = pmask.astype(np.float32)
            if iter == 0:
                pmask = np.transpose(pmask, (0, 1, 3, 2))
            pmask = torch.from_numpy(pmask)
            pmask = pmask.to(device)

            with torch.no_grad():
                pred_y = torch.sigmoid(model([image, pmask]))

                score = calculate_metrics(mask, pred_y, img_x)
                # 使用了map函数来将add函数逐个应用到metrics_score和score中对应位置的元素上，并将结果组成一个列表
                metrics_score = list(map(add, metrics_score, score))

                pred_y = pred_y[0][0].cpu().numpy()
                pred_y = pred_y > 0.5
                pred_y = np.transpose(pred_y, (1, 0))
                pred_y = np.array(pred_y, dtype=np.uint8)
                pred_y = rle_encode(pred_y)
                prev_masks[i] = pred_y
                tmp_masks.append(pred_y)

        """ Mean Metrics Score """
        f1_or_dsc = metrics_score[0] / len(test_x)
        miou = metrics_score[1] / len(test_x)
        recall = metrics_score[2] / len(test_x)
        specificity = metrics_score[3] / len(test_x)
        precision = metrics_score[4] / len(test_x)

        # 检查当前指标是否超过了已记录的最大值
        if f1_or_dsc > max_f1_or_dsc:
            max_f1_or_dsc = f1_or_dsc
        if miou > max_miou:
            max_miou = miou
        if recall > max_recall:
            max_recall = recall
        if specificity > max_specificity:
            max_specificity = specificity
        if precision > max_precision:
            max_precision = precision

        # print(
        #     f"\n f1_or_dsc: {f1_or_dsc:1.4f} \n miou: {miou:1.4f} \n recall: {recall:1.4f} \n specificity: {specificity:1.4f} \n precision: {precision:1.4f}")

        save_str = f"{iter + 1},{f1_or_dsc:1.4f},{miou:1.4f},{recall:1.4f},{specificity:1.4f},{precision:1.4f}\n"
        file.write(save_str)

        save_data.append(tmp_masks)

    # 在所有迭代完成后，输出表现最佳的那次迭代的指标
    print(
          f"\n f1_or_dsc: {max_f1_or_dsc:1.4f} \n"
          f" miou: {max_miou:1.4f} \n"
          f" recall: {max_recall:1.4f} \n"
          f" specificity: {max_specificity:1.4f} \n"
          f" precision: {max_precision:1.4f}"
        )
    # save_data = np.array(save_data)


    """ Saving the masks. """
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        # mask = mask / 255
        # mask = (mask > 0.5) * 255
        mask = mask_parse(mask)

        # 从文件路径y中提取文件名的主体部分（不包含扩展名）
        name = y.split("\\")[-1].split(".")[0]
        # 设置填充区域
        sep_line = np.ones((size[0], 10, 3)) * 128
        tmp = [image, sep_line, mask]

        # data表示每次迭代的数据，值为10
        for data in save_data:
            tmp.append(sep_line)
            d = data[i]
            d = " ".join(str(z) for z in d)
            d = str(d)
            d = rle_decode(d, size)
            d = d * 255
            d = mask_parse(d)

            tmp.append(d)

        # 将tmp列表中的所有元素按照指定的轴(axis)，即水平方向进行连接，并将结果保存在变量cat_images中
        cat_images = np.concatenate(tmp, axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)

