import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identify_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)       # 회귀는 항등함수, 이진분류는 시그모이드, 다분류는 소프트맥스를 주로 사용
    return y

# x = np.arange(-5.0, 5.0, 0.1)       # arange와 array를 혼돈하지 말 것
# y1 = step_function(x)
# y2 = sigmoid(x)
# y3 = relu(x)
# plt.plot(x, y1, label='step')
# plt.plot(x, y2, label='sigmoid', linestyle='--')
# plt.plot(x, y3, label='relu', linestyle='--')
# plt.ylim(-0.1, 1.1)
# plt.show()

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)                    # [0.31682708 0.69627909]

# 출력층을 만들어 보자
# 회귀는 항등함수, 이진분류는 시그모이드, 다분류는 소프트맥스를 주로 사용
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)       # 오버플로우 대비책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y                    # 원소 각각은 0 ~ 1 출력, 합은 1이 된다. 단조증가함수(학습할때만 계산, inference할때는 생략)





