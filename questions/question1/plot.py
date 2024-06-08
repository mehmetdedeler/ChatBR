import matplotlib.pyplot as plt
import numpy as np


def plot_hist():
    data = {
        "project": ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"],
        "OB": [0.7976, 0.8102, 0.8265, 0.8348, 0.8438, 0.8601],
        "OB_EB(OB)": [0.7880, 0.7850, 0.8228, 0.7982, 0.7994, 0.7746],
        "OB_SR(OB)": [0.7717, 0.8135, 0.8134, 0.8018, 0.8167, 0.8032],
        "EB": [0.7985, 0.7943, 0.8168, 0.8167, 0.8204, 0.8104],
        "OB_EB(EB)": [0.8067, 0.8195, 0.8385, 0.8122, 0.8307, 0.8402],
        "EB_SR(EB)": [0.7751, 0.7929, 0.8231, 0.8055, 0.8192, 0.8205],
        "SR": [0.7350, 0.7561, 0.8229, 0.7805, 0.8173, 0.7815],
        "OB_SR(SR)": [0.7669, 0.7524, 0.8038, 0.7692, 0.8113, 0.7653],
        "EB_SR(SR)": [0.7568, 0.7536, 0.8096, 0.7864, 0.8348, 0.7818]
    }

    # Reorganize data for plotting
    plot_data = []
    titles = data["project"]
    feature_list = ["OB", "OB_EB(OB)", "OB_SR(OB)", "EB", "OB_EB(EB)", "EB_SR(EB)", "SR", "OB_SR(SR)", "EB_SR(SR)"]

    for project in titles:
        project_values = []
        for feature in feature_list:
            if feature in data:
                project_values.extend(data[feature][titles.index(project):titles.index(project) + 1])
        plot_data.append(project_values)

    plt.rcParams["font.sans-serif"] = "times new roman"
    plt.rcParams["font.size"] = 10
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.tight_layout(pad=2.0)
    X = np.arange(3)
    width = 0.20
    colors = ['#7ea2ed', '#fcd3b5', '#df8d8f']

    for i in range(len(plot_data)):
        ax = axs[i // 3, i % 3]
        for j in range(3):
            bars = ax.bar(X + width * j, plot_data[i][j * 3: (j * 3) + 3], width=width, color=colors[j], label=f'num {j + 1}')
        ax.set_xticks([i*width+j for j in X for i in range(3)])
        ax.set_xticklabels(feature_list, rotation=90, fontsize=8)
        ax.tick_params(axis='x', which='both', labelbottom=True, pad=-10)  # 放置刻度标签在刻度线的上方
        ax.set_title(titles[i])

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 2.75), fancybox=True, shadow=True, ncol=3)
    plt.subplots_adjust(hspace=0.4, top=0.94)
    plt.show()
import matplotlib.pyplot as plt

# Data
projects = ['AspectJ', 'Birt', 'Eclipse', 'JDT', 'SWT', 'Tomcat']
OB = [0.7976, 0.8102, 0.8265, 0.8348, 0.8438, 0.8601]
OB_Avg = [0.7798, 0.7992, 0.8181, 0.8000, 0.8081, 0.7889]
EB = [0.7985, 0.7943, 0.8168, 0.8167, 0.8204, 0.8104]
EB_Avg = [0.7909, 0.8062, 0.8308, 0.8088, 0.8249, 0.8303]
SR = [0.7350, 0.7561, 0.8229, 0.7805, 0.8173, 0.7815]
SR_Avg = [0.7619, 0.7530, 0.8067, 0.7778, 0.8231, 0.7735]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Performance Metrics for Different Projects')

# Plotting each project
for i, project in enumerate(projects):
    row = i // 3
    col = i % 3
    axs[row, col].plot([OB[i], OB_Avg[i], EB[i], EB_Avg[i], SR[i], SR_Avg[i]], label=project)
    axs[row, col].set_title(project)
    axs[row, col].set_xticks(range(6))
    axs[row, col].set_xticklabels(['OB', 'OB(Avg)', 'EB', 'EB(Avg)', 'SR', 'SR(Avg)'])
    axs[row, col].legend()

# Adjust layout
plt.tight_layout()
plt.show()


