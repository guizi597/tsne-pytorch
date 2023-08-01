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

model_lst = ["FSRL", "FAAN", "CIAN", "MSRMN"]
task_relation = "producedby"

plt.figure(figsize=(12, 6))

for i, every in enumerate(model_lst):
    model_name = every
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_file", type=str, default="tasks/" + model_name + "_" + task_relation + "_X.txt", help="file name of feature stored")
    parser.add_argument("--y_file", type=str, default="tasks/" + model_name + "_" + task_relation + "_labels.txt", help="file name of label stored")

    opt = parser.parse_args()
    print("get choice from args", opt)
    x_file = opt.x_file
    y_file = opt.y_file
    X = np.loadtxt(x_file)
    labels = np.loadtxt(y_file).tolist()
    tsne = TSNE(n_components=2, learning_rate=200).fit_transform(X)

    # 使用PCA 进行降维处理
    pca = PCA().fit_transform(X)
    # 设置画布的大小
    plt.figure(figsize=(12, 6))
    cm = matplotlib.colors.ListedColormap(['#FFA500', 'g'])
    ax1 = plt.subplot(121)
    t_scatter = plt.scatter(tsne[:, 0], tsne[:, 1], 20, labels, cmap=cm)
    ax1.set_title(task_relation + "(" + model_name + ")")
    plt.legend(handles=t_scatter.legend_elements()[0], labels=["Negative", "Positive"])

    ax2 = plt.subplot(122)
    p_scatter = plt.scatter(pca[:, 0], pca[:, 1], 20, labels, cmap=cm)
    ax2.set_title(task_relation + "(" + model_name + ")")
    plt.legend(handles=t_scatter.legend_elements()[0], labels=["Negative", "Positive"])

    plt.show()
