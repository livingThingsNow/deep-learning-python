# Program mnist_basic.py
# Date 2017/01/21
# original
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
# read the MNIST data -  mnist.py dataset directory
# load_mnist (train, train label), (test, test label)
# normalize -True 0.0 ~ 1.0, False 0~255
# flatten - True 784(1), False 1x28x28 (3)
# one_hot_label True -1 ,  False - 7,2 etc 

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(x_train.shape) 
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
print(label)  #5
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)
