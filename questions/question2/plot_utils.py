import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_hist():
    # 设置画布颜色为 blue
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots()

    # y 轴数据
    data = [[15, 20, 25, 30],
            [12, 16, 20, 24],
            [3, 12, 15, 18]]

    X = np.arange(4)
    width = 0.25

    plt.bar(X + width * 0, data[0], color="#296073", width=width, label='A')
    plt.bar(X + width * 0, data[1], color="#3596B5", width=width, label="B")
    plt.bar(X + width * 0, data[2], color="#ADC5CF", width=width, label='C')

    # 添加文字描述
    W = [width * 0, width * 1, width * 2]  # 偏移量
    for i in range(3):
        for a, b in zip(X + W[i], data[i]):  # zip拆包
            plt.text(a, b, "%.0f" % b, ha="center", va="bottom")  # 格式化字符串，保留0位小数

    plt.xlabel("Group")
    plt.ylabel("Num")

    plt.legend()
    plt.show()


def plot_multi_figure():
    fig, ax = plt.subplots(3, 3, figsize=(6, 6))

    fig.text(0.5, 0, 'x', ha='center')
    fig.text(0, 0.5, 'y', va='center')

    x = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    sins = np.sin(x)
    coss = np.cos(x)

    ax[1][1].plot(x, sins, 'r', alpha=0.5, lw=0.5, ls='-', marker='+', label='sin')
    ax2 = ax[1][1].twinx()
    ax2.plot(x, coss, 'g', alpha=0.5, lw=0.5, ls='-', marker='+', label='cos')
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    plt.tight_layout()
    plt.show()


def project_len_info():
    # 防止乱码，当然你也可以从配置上设置，不过需要下载字体包，第二行是防止一些符号显示有问题
    plt.rcParams["font.sans-serif"] = "times new roman"
    plt.rcParams['axes.unicode_minus'] = False
    # 分辨率设置
    # plt.rcParams['figure.dpi'] = 500
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    data_list = []
    for project in project_list:
        data = pd.read_csv(f"../question2/predict_data/bert_new/{project}_len_info_new.csv")
        data_list.append(data['total'])
    # 使用一个嵌套的 for 循环来遍历 ax 的行和列
    for i in range(2):  # 行
        for j in range(3):  # 列
            # 使用索引来访问 ax 和 data_list 中的对应元素
            ax[i][j].bar(range(0, len(data_list[i * 3 + j])), data_list[i * 3 + j])
            # 在每个 ax 中添加水平线
            ax[i][j].axhline(4000, ls="--", c="red")
            # 在每个 ax 中添加 title
            ax[i][j].set_title(project_list[i * 3 + j], verticalalignment='bottom')
            # # 隐藏右边、上边的spine
            # ax[i][j].spines["right"].set_color("none")
            # ax[i][j].spines["top"].set_color("none")
    plt.tight_layout()
    plt.show()


def plot_llm_dataset():
    # 防止乱码，当然你也可以从配置上设置，不过需要下载字体包，第二行是防止一些符号显示有问题
    plt.rcParams["font.sans-serif"] = "times new roman"
    plt.rcParams['axes.unicode_minus'] = False
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    data_list, data_list_final = [], []
    for project in project_list:
        data = pd.read_csv(f"../question2/predict_data/bert_new/{project}_len_info_new.csv")
        data_list.append(len(data['total']))
        data_list_final.append(len([bug for bug in data['total'] if bug <= 4000]))

    X = np.arange(6)
    plt.bar(X, height=data_list, color="#ADC5CF", width=0.25)
    plt.bar(X, height=data_list_final, color="#3596B5", width=0.25)

    for i in range(6):
        a, b = data_list[i], data_list_final[i]
        plt.text(X[i] + 0.07, a, "%.0f" % a, ha="center", va="bottom")  # 格式化字符串，保留0位小数
        plt.text(X[i] - 0.07, b, "%.0f" % b, ha="center", va="bottom")  # 格式化字符串，保留0位小数

    plt.xlabel("Project")
    plt.ylabel("Bug Num")
    plt.show()


def plot_ring_pic(project_list, data):
    """绘制圆环图"""
    from math import pi
    from matplotlib.lines import Line2D
    plt.rcParams["font.sans-serif"] = "times new roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(2, 3, figsize=(10, 8), subplot_kw={'projection': 'polar'})
    colors = ['#4393E5', '#43BAE5', '#7AE6EA']
    xs = [[(i * pi * 2) / 600 for i in project_data] for project_data in data]
    ys = [0.8, 1.7, 2.6]
    left = (90 * pi * 2) / 360  # 控制起始位置

    # 在末尾标出线条和点来使它们变圆
    for i, x in enumerate(xs):
        row, col = i // 3, i % 3
        for j in range(3):
            ax[row][col].barh(ys[j], x[j], left=left, height=0.8, color=colors[j], label=data[i][j])
        ax[row][col].grid(False)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])
        ax[row][col].spines.clear()  # 清除每个子图的坐标轴
        ax[row][col].set_title(project_list[i])  # 添加标题
        # 省略legend_elements和handles参数，只调用legend()函数
        ax[row][col].legend(loc='lower center', bbox_to_anchor=(0.55, 0.16), frameon=False)

    legend = [
        Line2D([0], [0], marker='o', color='w', label='proper report', markerfacecolor=colors[0], markersize=16),
        Line2D([0], [0], marker='o', color='w', label='high-quality', markerfacecolor=colors[1], markersize=16),
        Line2D([0], [0], marker='o', color='w', label='raw report', markerfacecolor=colors[2], markersize=16)
    ]
    # 把这一行放在这里
    fig.legend(handles=legend, loc='lower center', shadow=True, fontsize=16, ncol=3)
    plt.tight_layout()
    plt.show()


def plot_llm_dataset_ring(predict_data_path, max_length, project_list):
    # 筛选项目中缺陷报告长度小于K的样本
    llm_dataset_info = []
    for project in project_list:
        csv_file_path = os.path.join(predict_data_path, "{}_len_info.csv".format(project))
        len_df = pd.read_csv(csv_file_path)
        llm_dataset_info.append([len(len_df[len_df['total'] < max_length]), len(len_df['bug_id']), 600])

    # 绘制圆环图
    plot_ring_pic(project_list, llm_dataset_info)


def plot_box_pic():
    """绘制语义相似度的箱图"""

    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

    for i, project in enumerate(project_list):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        df = pd.read_csv(f'../question4/similarity_result/prompt_0/{project}_similarity_result_word2vec.csv')
        for feat_idx, feature in enumerate(['OB', 'EB', 'SR']):
            group_df = df[df['feature_name'] == feature].groupby('feature_combine')
            for group_idx, (name, group) in enumerate(group_df):
                ax.boxplot(group["score"], positions=[(feat_idx+1)*3+group_idx], widths=[0.6], labels=[name])

        ax.set_title(project)
        ax.set_xlabel("feature")
        ax.set_ylabel("score")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_box_pic()

    # plot_hist()

    # project_len_info()

    # plot_llm_dataset()
    # projects = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    # plot_data = [
    #     [8, 10, 600],
    #     [59, 60, 600],
    #     [31, 31, 600],
    #     [45, 47, 600],
    #     [29, 31, 600],
    #     [12, 14, 600]
    # ]

    # plot_ring_pic(projects, plot_data)
