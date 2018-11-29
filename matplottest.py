#####################################################################
# matplottest.py
# pip install matplotlib    # 3.0.2 설치된다. (3.0.0 에서는 fond관련 파일을 찾을 수 없다는 에러 뜬다
#

import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1)    # 0에서 6까지 0.1 간격으로
y = np.sin(x)
# 그래프 그리기
# plt.plot(x, y)
# plt.show()

y1 = np.sin(x)
y2 = np.cos(x)
# 그래프 그리기
# plt.plot(x, y1, label="sin")
# plt.plot(x, y2, label="cos", linestyle="--")    # cos는 점선으로 그리자
# plt.xlabel("x")     # x축 이름
# plt.ylabel("y")     # y축 이름
# plt.title('sin & cos')      # 제목
# plt.legend()
# plt.show()

from matplotlib.image import imread
img = imread('lena.png')
plt.imshow(img)
plt.show()
