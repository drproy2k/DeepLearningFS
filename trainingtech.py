import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100)      # 100차원 벡터 1000개 생성. 가우시안분포를 갖는 값으로..
node_num = 100
hidden_layer_size = 5
activations = {}        # 여기에 활성화 결과를 저장할것이다
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    # w = np.random.randn(node_num, node_num) * 1       # 표준편차를 1로 정규화 --> 0과 1에 집중 --> 0과 1 부근에서 기울기=0, gradient vanishing
    # w = np.random.randn(node_num, node_num) * 0.01      # 표준편차를 0.01로 정규화 --> 0.5 부근에 집중 --> 표현력 제한
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)     # Xavier 사비에르 : 앞층의 노드수만큼 루트해서 나눔. 활성함수가 선형일때만 OK
                                                                    # sigmoid, tanh는 중앙 부근이 선형에 가까우므로 Xavier가 좋다
    # w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num) # He초기값 : ReLU에 좋다. ReLU는 반쪽이 0이니까 2배 넓게 확장
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z      # x와 같은 차원이다

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))     # 100차원 벡터를 단일 숫자화한거 1000개, 30 = x축 눈금수, range = x축 범위
plt.show()
