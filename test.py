import getpass
import sys
from numpy import *

from click._compat import raw_input
from gevent import os

from MachineLearning.KNN import KNN

"""print("hello world!")

name = "Alex Li"

name2 = name
print(name, name2)

name = "Jack"
print("name:"+name)
print("name2:"+name2)"""

"""name = input("What is your name?")
print("Hello " + name)
pwd = getpass.getpass("请输入密码：")
print(pwd)

print(sys.argv)
os.system(''.join(sys.argv[1:]))

name_list = ['alex', 'seven', 'eric']

name = "alex"
print("i am %s " % name)

person = {"name": "mr.wu", 'age': 18}

name = input('请输入用户名：')
pwd = getpass.getpass('请输入密码：')

if name == "wjy" and pwd == "123456":
    print("欢迎，alex！")
else:
    print("用户名和密码错误")"""
group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']
result_type = KNN.classify0([0,0],group,labels,3)
print(result_type)
