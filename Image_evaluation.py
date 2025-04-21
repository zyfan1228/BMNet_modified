import os
import math

from PIL import Image
import numpy as np
import cv2
from skimage import filters
from skimage.measure import shannon_entropy
from brisque import BRISQUE


##使用PIL库分割图像
def split_image(image_path, output_folder, name):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    # breakpoint()
    tile_w, tile_h = w // 3, h // 3  # 分割为3x3网格
    for i in range(3):
        for j in range(3):
            box = (j * tile_w, i * tile_h, (j + 1) * tile_w, (i + 1) * tile_h)
            region = img.crop(box)
            region.save(f"{output_folder}/{name}_tile_{i}_{j}.png")
    print("分割完成，子图保存在:", output_folder)


# 1. Tenengrad梯度法（Sobel算子）
def tenengrad(img_gray):
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return np.mean(gradient ** 2)  # 梯度平方均值


# 2. Laplacian梯度法
def laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()  # 二阶导数方差


# 3. Brenner梯度法
def brenner(img_gray):
    diff = np.diff(img_gray, axis=1)[:-1, :]  # 相邻像素水平差
    return np.sum(diff ** 2)  # 平方和


# 4. SMD/SMD2（灰度方差）
def smd(img_gray):
    diff_x = np.abs(np.diff(img_gray, axis=0))[:-1, :]
    diff_x = np.abs(np.diff(img_gray, axis=0))
    print(diff_x.shape)  #
    diff_y = np.abs(np.diff(img_gray, axis=1))[:, :-1]
    print(diff_y.shape)
    return np.sum(diff_x + diff_y)  # 水平和垂直差异和


def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0] - 1):
        for y in range(0, shape[1]):
            out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
    return out


def smd2(img_gray):
    diff_x = np.diff(img_gray, axis=0)[:-1, :]
    diff_y = np.diff(img_gray, axis=1)[:, :-1]
    return np.sum(diff_x * diff_y)  # 差异乘积


def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(int(img[x, y] - int(img[x, y + 1])))
    return out


# 5. 方差法
def variance(img_gray):
    return np.var(img_gray)  # 全局灰度方差


# 6. 信息熵法

def entropy(img_gray):
    return shannon_entropy(img_gray)  # 灰度分布熵值


# 7. BRISQUE（需安装brisque库）
# pip install brisque

def brisque_score(img_rgb):
    return BRISQUE().score(img_rgb)  # 基于NSS和SVM的分类评分


# 8. RankIQA/DIQA（需预训练模型）
# 示例代码框架（需PyTorch环境）
# def rankiqa_score(img_rgb):
#     model = load_pretrained_model()  # 加载预训练模型
#     img_tensor = preprocess(img_rgb)
#     return model.predict(img_tensor)  # 输出质量分数


# 9. 颜色分量加权法（模拟专利方法）
def color_weighted(img_rgb):
    # 提取RGB和亮度分量
    r, g, b = cv2.split(img_rgb)
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)
    # 子图像阈值分割与权重分配（自定义逻辑）
    weights = [0.4, 0.3, 0.2, 0.1]  # 示例权重
    return np.average([np.var(c) for c in [r, g, b, y]], weights=weights)


# fft统计高频成分的能量占比
def high_freq_energy_ratio(img_gray):
    # 傅里叶变换及中心化
    radius_ratio = 0.2
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # 计算幅度谱和能量谱
    magnitude = np.abs(fshift)
    energy = magnitude ** 2

    # 图像尺寸及中心坐标
    h, w = img_gray.shape
    cy, cx = h // 2, w // 2

    # 生成距离矩阵
    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)
    distance = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    # 定义高频区域掩模
    radius = min(cy, cx) * radius_ratio
    high_freq_mask = (distance > radius)

    # 计算高频能量和总能量
    total_energy = np.sum(energy)
    high_energy = np.sum(energy[high_freq_mask])

    return high_energy / total_energy * 100


def evaluate_all_blocks(image_path, output_folder, name='input'):
    # 1. 分割图像
    split_image(image_path, output_folder, name=name)
    # breakpoint()

    # 2. 遍历子图并评价
    results = []
    for i in range(3):
        for j in range(3):
            img_path = f"{output_folder}/{name}_tile_{i}_{j}.png"
            # breakpoint()
            img_gray = np.array(Image.open(img_path))
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            # img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # 调用各评价函数
            scores = {
                # 'Tenengrad': tenengrad(img_gray),
                # 'Laplacian': laplacian(img_gray),
                # 'Brenner': brenner(img_gray),
                # 'SMD': SMD(img_gray),
                # 'SMD2': SMD2(img_gray),
                # 'Variance': variance(img_gray),
                # 'Entropy': entropy(img_gray),
                'BRISQUE': brisque_score(img_rgb),
                # 'fft': high_freq_energy_ratio(img_gray)
                # 'ColorWeighted': color_weighted(img_rgb)
            }
            results.append((f'{name}_tile_{i}_{j}', scores))

    # 3. 输出结果
    for block, scores in results:
        print(f"--- {block} 清晰度评分 ---")
        for metric, value in scores.items():
            print(f"{metric}: {value:.2f}\n")
    print(results)



if __name__ == '__main__':
    # 调用示例
    evaluate_all_blocks(
        "/data3/fanzhuoyao/MY_WORKS/MyGithub/BMNet_modified/test_results/2025_04_21_09_04_00/output_idx_380.png", 
        "./test_results/image_evaluation_split",
        name="output")
