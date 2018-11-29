import numpy as np
x = np.array([1.0, 2.0, 3.0])       # [1. 2. 3.] <class 'numpy.ndarray'> float64
print(x, type(x), x.dtype)

x = np.asarray([1.0, 2.0, 3.0])     # 똑같다. [1. 2. 3.] <class 'numpy.ndarray'> float64
print(x, type(x), x.dtype)

y = np.array([2.0, 4.0, 6.0])
print(x - y)
print(x * y)

A = np.array([
    [1, 2],
    [3, 4]
])
print(A)
print(A.shape, A.dtype)
print(A * 10)

B = np.array([10, 20])
print(A * B)       # B가 종으로 브로드케스트 된 후, element-wise-product를 한다

B = B.reshape([2, 1])
print(B)
print(A * B)        # B가 횡으로 브로드케스트 된 후, element-wise-product를 한다

B = np.array([
    [5, 6],
    [7, 8]
])
print(A * B)           # [[5 12] [21 32]]  element-wise-product를 한다
print(np.dot(A, B))    # [[19 22] [43 50]] 내적을 구한다

# X = np.array([[51, 55], [14, 19], [0, 4]])
# print(X)
#
# for x in X:
#     print(x)
#
# X = X.flatten()
# print(X)
# print(X > 15)
# print(X[X > 15])        # 조건을 만족하는 원소만 뽑아서 리스트로 출력
# print(sum(X > 15))      # True == 1, output = 3

