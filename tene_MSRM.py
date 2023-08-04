# -*- coding: utf-8 -*-
# @Time : 2023/7/31 18:56
# @Author : zzu_nlp_gmy
# @File : tene_MSRM.py
# @File_Description :

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import time
from matplotlib.pyplot import MultipleLocator

cur_time = time.perf_counter()

lr = 200
init = 'warn'  # ['warn', 'random', 'pca']
method = 'exact'  # ['barnes_hut', 'exact']
# margin = 35
# lim = [-80, 80]
font_size = 40
node_size = 80
legend_size = 32
plot_only = 2200

my_x_ticks = np.arange(-70, 85, 35)
my_y_ticks = np.arange(-70, 85, 35)
print('lr:{}, init:{}'.format(lr, init))

font = dict(family='times new roman', size=40)

model_lst = ["FAAN"]
# model_lst = ["FSRL","CIAN"]
# task_relation = "teamcoach"
# task_relation = "animalsuchasinvertebrate"
task_relation = "producedby"

plt.figure(figsize=(256, 128))

plt.rcParams['figure.figsize'] = (25.6, 12.8)
ig, axs = plt.subplots(1, 2)
ax1, ax2 = axs.flatten()
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.axis('equal')
cm = matplotlib.colors.ListedColormap(['#FFA500', 'g'])

"""
1.第一张图
"""
model_name = model_lst[0]
parser = argparse.ArgumentParser()
parser.add_argument("--x_file", type=str, default="tasks/" + model_name + "_" + task_relation + "_X.txt", help="file name of feature stored")
parser.add_argument("--y_file", type=str, default="tasks/" + model_name + "_" + task_relation + "_labels.txt", help="file name of label stored")

opt = parser.parse_args()
print("get choice from args", opt)
x_file = opt.x_file
y_file = opt.y_file
X = np.loadtxt(x_file)
labels = np.loadtxt(y_file).tolist()[plot_only:]
tsne1 = TSNE(learning_rate=lr, init=init, method=method).fit_transform(X[plot_only:,:])

# 使用PCA 进行降维处理
# pca = PCA().fit_transform(X)

t_scatter = ax1.scatter(tsne1[:, 0], tsne1[:, 1], node_size, labels, cmap=cm)
ax1.set_title(task_relation + "(" + model_name + ")", fontsize=font_size, pad=30.0, fontproperties=font)
ax1.legend(handles=t_scatter.legend_elements()[0], labels=["Negative", "Positive"], loc="lower right", prop={'size': legend_size,'family':'times new roman'})

# # 把x轴的刻度间隔设置为25
# x_major_locator = MultipleLocator(margin)
# # 把y轴的刻度间隔设置为25
# y_major_locator = MultipleLocator(margin)
# # 把x轴的主刻度设置为25的倍数
# ax1.xaxis.set_major_locator(x_major_locator)
# # 把y轴的主刻度设置为25的倍数
# ax1.yaxis.set_major_locator(y_major_locator)
# # 把x轴的刻度范围设置为-60到60，因为不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# ax1.set_xlim(lim[0], lim[1])
# ax1.set_xlim(lim[0], lim[1])
ax1.set_xticks(my_x_ticks)
ax1.set_yticks(my_y_ticks)
ax1.set_xticklabels(my_x_ticks, fontdict=font)
ax1.set_yticklabels(my_y_ticks, fontdict=font)


# 设置刻度的参数
ax1.tick_params(labelsize=font_size, length=10)



"""
2.第二张图
"""
model_name = model_lst[0]
parser = argparse.ArgumentParser()
parser.add_argument("--x_file", type=str, default="MSRM_tasks/" + model_name + "_" + task_relation + "_X.txt", help="file name of feature stored")
parser.add_argument("--y_file", type=str, default="MSRM_tasks/" + model_name + "_" + task_relation + "_labels.txt", help="file name of label stored")

opt = parser.parse_args()
print("get choice from args", opt)
x_file = opt.x_file
y_file = opt.y_file
X = np.loadtxt(x_file)
labels = np.loadtxt(y_file).tolist()[plot_only:]
tsne2 = TSNE(learning_rate=lr, init=init).fit_transform(X[plot_only:,:])

# 使用PCA 进行降维处理
# pca = PCA().fit_transform(X)

t_scatter = ax2.scatter(tsne2[:, 0], tsne2[:, 1], node_size, labels, cmap=cm)
ax2.set_title(task_relation + "(" + model_name + " + MSRM" + ")", fontsize=font_size, pad=30.0, fontproperties=font)
ax2.legend(handles=t_scatter.legend_elements()[0], labels=["Negative", "Positive"], loc="lower right", prop={'size': legend_size,'family':'times new roman'})

# # 把x轴的刻度间隔设置为25
# x_major_locator = MultipleLocator(margin)
# # 把y轴的刻度间隔设置为25
# y_major_locator = MultipleLocator(margin)
# # 把x轴的主刻度设置为25的倍数
# ax2.xaxis.set_major_locator(x_major_locator)
# # 把y轴的主刻度设置为25的倍数
# ax2.yaxis.set_major_locator(y_major_locator)
# # 把x轴的刻度范围设置为-60到60，因为不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# ax2.set_xlim(lim[0], lim[1])
# ax2.set_xlim(lim[0], lim[1])
ax2.set_xticks(my_x_ticks)
ax2.set_yticks(my_y_ticks)
ax2.set_xticklabels(my_x_ticks, fontdict=font)
ax2.set_yticklabels(my_y_ticks, fontdict=font)

# 设置刻度的参数
ax2.tick_params(labelsize=font_size, length=10)
plt.savefig(r"img_0804/" + model_lst[0] + '_' + task_relation + "_0804.png", dpi=100, bbox_inches="tight")
plt.show()
print("Complete! Consume time is {}".format(time.perf_counter() - cur_time))