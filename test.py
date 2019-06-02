import getpass
import sys
from numpy import *
from click._compat import raw_input
from gevent import os
import numpy as np
import pandas as pd
from MachineLearning.KNN import KNN
import matplotlib as mpl
import matplotlib.pyplot as plt

'''group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']
result_type = KNN.classify0([0,0],group,labels,3)
print(result_type)'''

datingDataMat, datingDataLabel = KNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingDataLabel[0:20])

fig = plt.figure()
#add_subplot(mnp)添加子轴、图。subplot（m,n,p）或者subplot（mnp）此函数最常用：subplot是将多个图画到一个平面上的工具。
# 其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，如果第一个数字是2就是表示2行图。
# p是指你现在要把曲线画到figure中哪个图上，最后一个如果是1表示是从左到右第一个位置。
ax = fig.add_subplot(111)#将画布分割成1行1列，图像画在从左到右从上到下的第1块
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],30*array(datingDataLabel),30*array(datingDataLabel))
ax.set_title('Scatter Plot')
plt.legend('x')
plt.show()