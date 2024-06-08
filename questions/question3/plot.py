import matplotlib.pyplot as plt
import numpy as np


def plot():
    data = [
        [[7, 0, 0], [4, 3, 0], [7, 0, 0], [7, 0, 0], [7, 0, 0], [5, 2, 0]],
        [[55, 3, 0], [42, 15, 1], [54, 4, 0], [36, 19, 1], [38, 17, 2], [37, 19, 2]],
        [[28, 1, 0], [23, 6, 0], [25, 3, 1], [22, 6, 1], [26, 3, 0], [22, 6, 1]],
        [[40, 2, 0], [29, 12, 0], [33, 8, 1], [35, 6, 1], [39, 3, 0], [31, 9, 2]],
        [[24, 0, 1], [20, 5, 0], [19, 6, 0], [20, 3, 2], [21, 4, 0], [14, 11, 0]],
        [[9, 1, 0], [8, 2, 0], [7, 3, 0], [9, 1, 0], [10, 0, 0], [7, 2, 1]]
    ]
    plt.rcParams["font.sans-serif"] = "times new roman"
    plt.rcParams["font.size"] = 10
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.tight_layout(pad=2.0)
    titles = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    feature_list = ['OB', 'EB', 'SR', 'OB_EB', 'OB_SR', 'EB_SR']
    X = np.arange(6)
    width = 0.20  # 调整每个条形的宽度
    colors = ['#7ea2ed', '#fcd3b5', '#df8d8f']

    for i in range(len(data)):
        ax = axs[i // 3, i % 3]
        for j in range(3):
            bars = ax.bar(X + width * j, [d[j] for d in data[i]], width=width, color=colors[j], label=f'num {j + 1}')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')
            # 隐藏右边、上边的spine
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
        ax.set_xticks(X + width)
        ax.set_xticklabels(feature_list)  # 设置 x 轴标签为特征列表
        ax.set_title(titles[i])

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 2.75), fancybox=True, shadow=True, ncol=3)  # 创建一个公共图例
    plt.subplots_adjust(hspace=0.4, top=0.88)  # 调整间距和顶部空间
    plt.show()


def plot_1():
    plt.rcParams["font.sans-serif"] = "times new roman"
    plt.rcParams["font.size"] = 12
    plt.rcParams['axes.unicode_minus'] = False
    data = [[434, 74, 4], [386, 111, 13]]
    bar_width = 0.20
    X = range(len(data[0]))  # 生成X轴坐标
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    for i, row in enumerate(data):
        bars = ax.barh([x + i * bar_width for x in X], row, bar_width, label=f"miss {i+1} element")
        for bar, value in zip(bars, row):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, value,
                    ha='left', va='center')

    ax.set_yticks(range(0, 3, 1))
    ax.set_yticklabels(["num 1", 'num 2', 'num 3'])
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # 添加x轴方向的虚线网格线
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
