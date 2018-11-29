###################################################################
# backpropagation.py
# Back Propagation을 이용한 미분 방법
#

# 두 입력을 단순히 곱해서 다음단에 전달하는 mul layer
# 이 경우, 개개의 입력 변수에 대한 편미분은, 즉, backpropagation을 하는 경우,
# 윗단으로 부터 전달 받은 값에 입력 값 두개를 서로 바꾸어 곱하면 된다
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y      # x와 y를 바꾼다
        dy = dout * self.x
        return dx, dy

# 두 입력을 단순히 더하는 레이어
# 이 경우, 개개의 입력 변수에 대한 편미분은, 즉, backpropagation을 하는 경우,
# 윗단으로 부터 전달 받은 값을 그대로 아래로 전달하면 된다
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# 예제1 : 100원짜리 사과 2개를 샀는데 부가세가 10%라면 총 금액을 계산하는 네트웍
apple = 100
apple_num = 2
tax = 1.1
# 레이어 설계
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)      # 220
# 역전파
dprice = 1      # delta_price / delta_price = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)     # 2.2  110  200

# 예제2 : 사과2개 귤3개를 구입하는 경우
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
# 계층설계
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(price)        # 715
print(dapple, dapple_num, dorange, dorange_num, dtax)       # 2.2  110  3.3  165  650
