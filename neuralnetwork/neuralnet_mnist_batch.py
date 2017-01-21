# coding: utf-8
# Program mnist_batch.py
# Date 2017/01/21
# original
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
# sample_weight.pkl重みとバイアスのパラメータがディクショナリ型の変数として保存
# np.argmax(x) 引数x に与えられた配列で最大の値を持つ要素のインデツクスを取得する
# 予測した答えと正解ラベルを比較して正解した割合を認識精度とする。
# normalize true 正規化



import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポート
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    print(w1.shape)
    print(w2.shape)
    print(w3.shape)


    return y


x, t = get_data()
network = init_network()
print(x.shape)	#(1000, 784)
print(x[0].shape)	#(784,)

batch_size = 100 # バッチの数
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
