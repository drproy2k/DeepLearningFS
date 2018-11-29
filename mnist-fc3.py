import sys, os
import numpy as np
from PIL import Image
sys.path.append(os.pardir)      # 부모 디렉토리의 파일을 가져올려고 설정
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 각 데이터 형상을 보자..
print(x_train.shape)        # (60000, 784)
print(t_train.shape)        # (60000,)
print(x_test.shape)         # (10000, 784)
print(t_test.shape)         # (10000,)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)        # 5

print(img.shape)    # (784,)
img = img.reshape(28, 28)
print(img.shape)    # (28, 28)

#img_show(img)      # 숫자 5 이미지를 출력한다

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

import pickle
def init_network():
    with open("sample_weight.pkl", 'rb') as f:      # 학습된 가중치 파라미터가 딕셔너리 형태로 저장되어 있다
        network = pickle.load(f)
    return network

from common.functions import sigmoid, softmax
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()

# 이미 학습된 가중치(모델)로 테스트를 해 보자
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)        # 확률이 가장 높은 원소의 인텍스를 리턴
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# 배치처리 하는 방법
x, t = get_data()
network = init_network()

batch_size = 100        # 배치 크기
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])      # True이면 1이되고, 이것들을 모두 더한다

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# 이제 학습해 보자
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # 중복을 허용해서 batch_size개수만큼 뽑는다
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 정답 레이블이 onehot인 경우,
def cross_entropy_error(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

# 정답 레이블이 숫자인 경우,
def cross_entropy_error_num(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arrange(batch_size), t])) / batch_size


