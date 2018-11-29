############################################################
# lossfunction.py
# 로스함수의 깊은 곳
#
import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    #print(y.shape)
    #input()     # (10, ), (2, 10) 이런식으로 나온다. (10, )를 (1, 10) 형태로 바꾸어야 한다.
    if y.ndim == 1:     # 입력 데이터가 달랑 하나인 경우,
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7        # log(0)를 방지하기 위해 아주 작은 값을 추가
    #print(t * np.log(y + delta))       # 10차원짜리 결과 벡터를 batch_size개수만큼 갖는 np.array가 생긴다
    #input()
    return -np.sum(t * np.log(y + delta)) / batch_size      # 10차원 x batch_size np.array 내의 모든 요소를 더한다

def cross_entropy_error_num(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size   # t = [2, 7, 0, 9, 4]라면  y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]
                                                                                # 즉, 정답 레이블에 해당하는 신경망 출력만 추출

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))     # 0.09750000000000003
print(cross_entropy_error(np.array(y), np.array(t)))    # 0.510825457099338
t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
y = [[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.3]]
print(mean_squared_error(np.array(y), np.array(t)))     # 0.5975
print(cross_entropy_error(np.array(y), np.array(t)))    # 2.302584092994546

# 데이터가 너무 많으면, 손실값의 전체 크기가 한도를 넘으니깐 미니배치로 나누어 처리하자
import sys, os
sys.path.append(os.pardir)      # 현 폴더위치를 부모폴더로 이동
import numpy as np
from dataset.mnist import load_mnist    # 부모폴터 밑에 dataset 폴더내의 mnist.py 안에 load_mist function을 로드

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)        # (60000, 784)
print(t_train.shape)        # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
#print(batch_mask)           # [ 2175 24588  7130 16284 40331  9015 32284 21406  2935  5001]
x_batch = x_train[batch_mask]   # 상기 순번의 입력데이터 10개가 뽑힌다
t_batch = t_train[batch_mask]
#print(t_batch)


# 손실함수의 기울기를 구해보자
# 수치미분 (반대말은 해석적미분)
def numerical_diff(f, x):   # mathod name과 입력 데이터 x를 입력으로 받는다
    h = 1e-4        # 0.0001    float32를 쓸 경우, 좋은 결과가 나온다
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

import matplotlib.pylab as plt
x = np.arange(0.0, 20.0, 0.1)       # 0에서 20까지 0.1 간격의 배열 x를 만든다
y = function_1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y)
#plt.show()
print(numerical_diff(function_1, 5))        # 0.1999999999990898  mathod name과 입력 데이터를 인자로 전달
print(numerical_diff(function_1, 10))       # 0.2999999999986347

# 편미분 예제
def function_2(x):
    return x[0]**2 + x[1]**2

# [3, 4]에서 x0에 대해 편미분
def function_2_div_x0(x0):
    return x0**2 + 4**2

# [3, 4]에서 x1에 대해 편미분
def function_2_div_x1(x1):
    return 3**2 + x1**2

print(numerical_diff(function_2_div_x0, 3.0))       # 6.00000000000378
print(numerical_diff(function_2_div_x1, 4.0))       # 7.999999999999119

# 기울기 구하기
def numerical_gradient(f, x):
    h = 1e-4    # 0.0001
    grad = np.zeros_like(x)     # x와 형상이 같은 배열을 생성
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val    # 값 복원
    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))     # [6. 8.]
print(numerical_gradient(function_2, np.array([0.0, 2.0])))     # [0. 4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0])))     # [6. 0.]


# 경사 하강법 구현하기
def gradient_descent(f, init_x, lr=0.01, step_num=100):     # methon name, init_x, learning rage, step_num
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

# learning rate을 적당히 잘 골라야...
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))        # [-6.11110793e-10  8.14814391e-10] ==> 0 근처로 간다
# learning rate이 너무 크면,
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))       # [-2.58983747e+13 -1.29524862e+12] 발산한다
# learning rate이 너무 작으면,
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))      # [-2.99999994  3.99999992] 거의 진척이 이루어지지 않는다

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)      # 평균 0, 표준편차 1인 가우시안 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):       # x : 입력 데이터, t : 레이블
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = SimpleNet()
print(net.W)        # [[ 0.63887628  1.40111608 -0.10539746], [ 0.44421827 -0.19655748  0.3820261 ]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)            # [0.78312221 0.66376791 0.28058502]
print(np.argmax(p))         # 0
t = np.array([0, 0, 1])
print(net.loss(x, t))       # 1.4158178017981566

def f(W):                   # W는 더미로 만든 것임. numerical_gradient에서 f(x)를 실행하는데, 일관성을 위해 f(W)를 정의
    return net.loss(x, t)   # main에서 net 인스턴스를 호출

#f = lambda W: net.loss(x, t)    # 위의 함수와 같은 의미
dW = numerical_gradient(f, net.W)       # main에서의 net 인스턴스를 호출
print(dW)                               # [[ 0.03379284  0.26377881 -0.29757165], [ 0.05068926  0.39566821 -0.44635747]]
                                        # w11이 0.03이므로 w11을 h만큼 늘리면, 손실함수 값은 0.03h만큼 증가한다는 의미
