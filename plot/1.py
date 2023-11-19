# 设置字体
import matplotlib as mpl
# mpl.rcParams['axes.unicode_minus'] = False
# mpl.rcParams['font.sans-serif'] = ['SimHei']#黑体
import matplotlib.pyplot as plt
import pylab as pl

from matplotlib import pyplot as plt

plt.subplot(2,3,1)
a = ["NCBI-disease", "BC2GM", "BC5CDR-Chem"]
x_14 = [0, 0.55, 1.1]
x_15 = [0.1, 0.65, 1.2]
x_16 = [0.2, 0.75, 1.3]
x_17 = [0.3, 0.85, 1.4]

# F1
b_14 = [0.8939,0.8470,0.9324] #biobert mrc
b_15 = [0.9018,0.8516,0.9376] #no reg
b_16 = [0.9021,0.8579,0.9391] #no prefix
b_17 = [0.9075,0.8610,0.9422] #pr

# 绘条形图
plt.bar(x_14, b_14, width=0.1, color='orange', label='BioBert+MRC')
plt.bar(x_15, b_15, width=0.1, color='silver', label='PS-MRC no regularization')
plt.bar(x_16, b_16, width=0.1, color='plum', label='PS-MRC no prefix')
plt.bar(x_17, b_17, width=0.1, color='cornflowerblue', label='PS-MRC')

import pylab as pl

# pl.xlabel('BioNER Dataset')  # make axis labels
pl.ylabel('F 1')
pl.ylim(0.845, 0.945)
# 设置坐标轴
x1 = [0.15, 0.7, 1.25]
plt.xticks(x1, a)  # 分组标签
# 设置图例
plt.legend()

plt.subplot(2,3,2)
a = ["NCBI-disease", "BC2GM", "BC5CDR-Chem"]
x_14 = [0, 0.55, 1.1]
x_15 = [0.1, 0.65, 1.2]
x_16 = [0.2, 0.75, 1.3]
x_17 = [0.3, 0.85, 1.4]

# P
b_14 = [0.8929,0.8559,0.9389] #biobert mrc
b_15 = [0.8944,0.8600,0.9492] #no reg
b_16 = [0.9064,0.8674,0.9415] #no prefix
b_17 = [0.9005,0.8725,0.9460] #pr
# 绘条形图
# plt.bar(range(len(a)), b_14, width=0.1, color='cornflowerblue')
plt.bar(x_14, b_14, width=0.1, color='orange')
plt.bar(x_15, b_15, width=0.1, color='silver')
plt.bar(x_16, b_16, width=0.1, color='plum')
plt.bar(x_17, b_17, width=0.1, color='cornflowerblue')

import pylab as pl

# pl.xlabel('BioNER Dataset')  # make axis labels
pl.ylabel('Precision')
pl.ylim(0.85, 0.955)
# 设置坐标轴
x2 = [0.15, 0.7, 1.25]
plt.xticks(x2, a)  # 分组标签
# 设置图例

plt.subplot(2,3,3)
a = ["NCBI-disease", "BC2GM", "BC5CDR-Chem"]
x_14 = [0, 0.55, 1.1]
x_15 = [0.1, 0.65, 1.2]
x_16 = [0.2, 0.75, 1.3]
x_17 = [0.3, 0.85, 1.4]

# R
b_14 = [0.8948,0.8382,0.9259] #biobert mrc
b_15 = [0.9093,0.8433,0.9263] #no reg
b_16 = [0.8979,0.8486,0.9366] #no prefix
b_17 = [0.9145,0.8498,0.9383] #pr
# 绘条形图
# plt.bar(range(len(a)), b_14, width=0.1, color='cornflowerblue')
plt.bar(x_14, b_14, width=0.1, color='orange')
plt.bar(x_15, b_15, width=0.1, color='silver')
plt.bar(x_16, b_16, width=0.1, color='plum')
plt.bar(x_17, b_17, width=0.1, color='cornflowerblue')

import pylab as pl

# pl.xlabel('BioNER Dataset')  # make axis labels
pl.ylabel('Recall')
pl.ylim(0.83, 0.944)
# 设置坐标轴
x3 = [0.15, 0.7, 1.25]
plt.xticks(x3, a)  # 分组标签

plt.show()
