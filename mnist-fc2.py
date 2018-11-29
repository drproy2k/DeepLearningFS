import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

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

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)       # (784, 100)
print(net.params['b1'].shape)       # (100, )
print(net.params['W2'].shape)       # (100, 10)
print(net.params['b2'].shape)       # (10, )

# x = np.random.rand(100, 784)    # 0 ~ 1 사이 균일데이터 100x784 생성, 더미 입력 데이터 100장 분량
# y = net.predict(x)
#print(y)
# t = np.random.rand(100, 10)
# grads = net.numerical_gradient(x, t)    # 기울기 계산
# print(grads['W1'].shape)       # (784, 100)
# print(grads['b1'].shape)       # (100, )
# print(grads['W2'].shape)       # (100, 10)
# print(grads['b2'].shape)       # (10, )

##############################################################################
# 미니배치 학습 구현하기
#
from dataset.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 하이퍼파라미터
iters_num = 10000               # 반복횟수
train_size = x_train.shape[0]   # 입력 데이터 개수
batch_size = 100                # 미니배치 크기
learning_rate = 0.1             # 학습 속도

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1 epoch 당 반복회수
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

counter = 0
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)     # 시간이 무척 오래 걸린다
    #grad = network.gradient(x_batch, t_batch)              # 성능 개선판

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 epoch당 정확도를 계산해 보자..
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        counter += 1
        print("train acc, test acc : ", train_acc, test_acc)

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


