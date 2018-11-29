# perceptron
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp > theta:
        return 1
    else:
        return 0
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

# numpy로 하면
import numpy as np
def AND_np(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND_np(0, 0))
print(AND_np(0, 1))
print(AND_np(1, 0))
print(AND_np(1, 1))
