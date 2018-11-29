###########################################################################
# layers.py
# Relu와 Sigmoid, Affine, SoftmaxWithLoss layer 관련 forward, backward를 구현한 것
#
import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):       # x는 단변수이거나 numpy
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0      # 0보다 작은 요소들은 0으로 처리
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

# FC layer
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)      # element단위로 sum
        return dx

# dy = np.array([[1, 2, 3], [4, 5, 6]])
# db = np.sum(dy, axis=0)     # [5 6 7]   0번째 축 = element단위
# db = np.sum(dy, axis=1)     # [6 15]    1번째 축 = 벡터단위
# db = np.sum(dy)             # 21        모든 element

# Softmax with Loss : 역전파를 간단히 처리할 수 있어서 loss를 포함해서 설계한다
import sys, os
sys.path.append(os.pardir)
from common.functions import *
class SoftmaxWithLOss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

from common.gradient import numerical_gradient
from collections import OrderedDict     # 순서가 있는 사전. 기본 포함 외장 함수
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 계층 생성
        self.layers = OrderedDict()     # 순서가 있는 dict
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])    # Affine 클라스를 dict에 저장
        self.layers['Relu1'] = Relu()                                           # Relu 클라스를 dict에 저장
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLOss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])     # x.shape[0] = 입력데이터 개수
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        # 재귀호출이 아니다. 외부 mathod를 호출한다.
        # numerical_gradient는 시간이 오래 걸런다. gradient를 사용하면 빠르다
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])     # W1 값에서 편미분. W1은 계속 변하고, 변경된 위치에서 편미분...
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])     # 기울기 W1은 정확치 않기때문에 수치미분하면 loss가 생기고,
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])     # W1 위치에서 편미분해서 learning rate를 곱한거 만큼 W1을 업데이트,
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])     # 그런후에 또 loss를 구하고.. loss가 점점 줄어든다..
        return grads

    def gradient(self, x, t):
        # 순전파 (Relu의 경우 순전파 결과에 의해 backpropagation이 영향을 받는다)
        self.loss(x, t)
        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())     # key(layer name)는 두고, value(layer class)만 리스트로 출력
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # 결과저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# Momentum : 속도 v를 도입해서 지그재그 방지
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# AdaGrad : 변화가 큰 가중치 element는 학습속도를 줄인다. 지그재그를 막아준다. 그러나 편향보정이 안되는 단점이 발생
# 편향보정이란? 학습이 아래방향으로 가다가 방향을 틀어서 위로 갈 수도 있게 해 주는 것(지그재그)
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * 1 / (np.sqrt(self.h[key]) + 1e-7) * grads[key]

# Adam : Momentum과 AdaGrad를 합친것. AdaGrad의 편향보정 안되는 단점 해결
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


# 오차역전파를 제대로 구현했는지 수치미분과 비교해 보자
from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)
print(grad_numerical)
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key, diff)        # W2 5.424260022447009e-09, b2 1.401669450826204e-07, W1 4.233750559003307e-10, b1 2.6204003077653688e-09
                            # 거의 차이가 없다

# 오차역전파를 이용한 학습
iters_num = 10000
train_size = x_train.shape[0]
print('train_size: ', train_size)
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)
counter = 0
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 오차역전파 기법으로 기울기 구하기
    grad = network.gradient(x_batch, t_batch)
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        counter += 1
        print(i, train_acc, test_acc)

# 결과를 담은 train, test 리스트를 그림으로 그려보자
# 두 그래프가 비슷하게 움직이면 오버피팅 문제가 잘 해결된 것이다
import matplotlib.pyplot as plt
x = list(np.arange(0, counter, 1))
plt.plot(x, train_acc_list, label='train_acc')
plt.plot(x, test_acc_list, label='test_acc', linestyle='--')
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend()        # 도표설명
plt.show()

