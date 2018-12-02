################################################################
# rnn.py
# Env : mp35Envs
# Note : pip install tensorflow로 설치함
#

import numpy as np

class MultiplyGate:
    def forward(self, W, x):
        return np.dot(W, x)

    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx

class AddGate:
    def forward(self, x1, x2):
        out = x1 + x2
        return out

    def backward(self, x1, x2, dout):
        dx1 = dout * np.ones_like(x1)
        dx2 = dout * np.ones_like(x2)
        return dx1, dx2

class Sigmoid:
    def forward(self, x):
        out = 1.0 / (1.0 + np.exp(-x))
        return out

    def backward(self, x, dout):
        out = self.forward(x)
        dx = (1.0 - out) * out * dout
        return dx

class Tanh:
    def forward(self, x):
        out = np.tanh(x)
        return out

    def backward(self, x, dout):
        out = self.forward(x)
        dx = (1.0 - np.square(out)) * dout
        return dx

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])        # y는 정답 클라스의 인덱스. probs 원소들 내에서 해당 인덱스 위치의 값만 리턴

    def diff(self, x, y):
        probs = self.predict(x)         # 개개의 클라스들의 확률값을 갖는 출력 벡터
        probs[y] -= 1.0                 # 정답 클라스 엘레멘트의 확률값만 -1해 준다. 어짜피 나머지는 0으로 처리되니까...
        return probs

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return dprev_s, dU, dW, dV

rnnLayer = RNNLayer()
