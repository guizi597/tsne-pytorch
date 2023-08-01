# -*- coding: utf-8 -*-
# @Time : 2023/5/8 14:39
# @Author : zzu_nlp_gmy
# @File : tsne.py
# @File_Description :

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

from matplotlib.pyplot import MultipleLocator

model_lst = ["FSRL", "FAAN", "CIAN", "MSRMN"]
# model_lst = ["FSRL","CIAN"]
task_relation = "animalsuchasinvertebrate"

plt.figure(figsize=(256, 256))
plt.savefig(r"images/"+"animalsuchasinvertebrate_ALL0728.png",dpi=100,bbox_inches="tight")
plt.rcParams['figure.figsize']=(25.6, 25.6)
ig, axs = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.2, hspace=0.3)
cm = matplotlib.colors.ListedColormap(['#FFA500', 'g'])
for i, ax in enumerate(axs.flat):
    model_name = model_lst[i]
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_file", type=str, default="tasks/" + model_name + "_" + task_relation + "_X.txt", help="file name of feature stored")
    parser.add_argument("--y_file", type=str, default="tasks/" + model_name + "_" + task_relation + "_labels.txt", help="file name of label stored")

    opt = parser.parse_args()
    print("get choice from args", opt)
    x_file = opt.x_file
    y_file = opt.y_file
    X = np.loadtxt(x_file)
    labels = np.loadtxt(y_file).tolist()
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(X)

    # 使用PCA 进行降维处理
    # pca = PCA().fit_transform(X)

    t_scatter = ax.scatter(tsne[:, 0], tsne[:, 1], 80, labels, cmap=cm)
    ax.set_title(task_relation + "(" + model_name + ")", fontsize=40)
    ax.legend(handles=t_scatter.legend_elements()[0], labels=["Negative", "Positive"], loc="lower right", prop={'size': 32})

    # 把x轴的刻度间隔设置为25
    x_major_locator = MultipleLocator(25)
    # 把y轴的刻度间隔设置为25
    y_major_locator = MultipleLocator(25)
    # 把x轴的主刻度设置为25的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    # 把y轴的主刻度设置为25的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把x轴的刻度范围设置为-60到60，因为不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    ax.set_xlim(-60, 60)
    ax.set_xlim(-60, 60)
    ax.tick_params(labelsize=40)
    # ax2 = plt.subplot(122)
    # p_scatter = plt.scatter(pca[:, 0], pca[:, 1], 20, labels, cmap=cm)
    # ax2.set_title(task_relation + "(" + model_name + ")")
    # plt.legend(handles=t_scatter.legend_elements()[0], labels=["Negative", "Positive"])
plt.show()
