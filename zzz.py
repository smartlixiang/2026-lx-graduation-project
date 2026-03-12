import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100


def main():
    # 数据目录
    data_root = "./data"

    # 读取 CIFAR-100 训练集
    dataset = CIFAR100(root=data_root, train=True, download=False)

    # CIFAR-100 类别名称
    class_names = dataset.classes

    # 取前 5 个类别（标签 0,1,2,3,4）
    selected_classes = [5, 6, 7, 8, 9]

    # 每个类别收集前 5 张图片
    images_per_class = 5
    class_to_images = {cls_idx: [] for cls_idx in selected_classes}

    for img, label in dataset:
        if label in class_to_images and len(class_to_images[label]) < images_per_class:
            class_to_images[label].append(np.array(img))

        # 如果所有类别都收集满了，就提前结束
        if all(len(class_to_images[cls_idx]) == images_per_class for cls_idx in selected_classes):
            break

    # 检查是否收集成功
    for cls_idx in selected_classes:
        if len(class_to_images[cls_idx]) < images_per_class:
            raise RuntimeError(
                f"类别 {cls_idx} ({class_names[cls_idx]}) 只找到 {len(class_to_images[cls_idx])} 张图片，不足 {images_per_class} 张。"
            )

    # 创建画布：5 行 5 列图像
    n_rows = len(selected_classes)
    n_cols = images_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    # 如果 axes 不是二维数组，强制转成二维
    axes = np.array(axes).reshape(n_rows, n_cols)

    # 绘制图片
    for row, cls_idx in enumerate(selected_classes):
        for col in range(n_cols):
            axes[row, col].imshow(class_to_images[cls_idx][col])
            axes[row, col].axis("off")

    # 调整布局：
    # left 留出左侧空白给标签
    plt.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0.05)

    # 在左侧空白区域添加类别标签
    for row, cls_idx in enumerate(selected_classes):
        # 获取这一行最左侧子图的位置
        ax = axes[row, 0]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2

        fig.text(
            0.08,                  # 左侧空白区域中的横坐标
            y_center,              # 与该行中心对齐
            class_names[cls_idx],  # 类别名称
            ha="center",
            va="center",
            fontsize=24
        )

    # 保存图片
    output_path = "figure.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"图像已保存到: {output_path}")


if __name__ == "__main__":
    main()
