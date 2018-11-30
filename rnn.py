################################################################
# rnn.py
# Env : mp35Envs
# Note : pip install tensorflow로 설치함
#

import os, re, io
import requests
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import tensorflow as tf

epochs = 20
batch_size = 250
max_sequence_length = 25        # 단어수
rnn_size = 10
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
dropout_keep_prob = tf.placeholder(tf.float32)

data_dir = 'temp'
data_file = 'text_data.txt'     # SMS 문자 데이터
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]
# 여기까지 text_data 리스트 준비과정
#print(text_data[0:4])                                               # ['ham\tGo until....\n', 'ham\tOk lar...\n', ... ]
text_data = [x.split('\t') for x in text_data if len(x) >= 1]      # [['ham', 'Go until....'], ['ham', 'Ok lar...'], ... ]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]    # label(target)과 data(train)를 커다란 두개의 리스트로 묶음
#print(text_data[0:4])
#print(len(text_data))      # 5574

# 소문자변환, 특수 문자 및 부가 공백 제거 등 정규화 수행
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string
text_data_train = [clean_text(x) for x in text_data_train]

# 텐서플로에 내장된 어휘 처리 함수로 문제 데이터를 처리 --> 문자 데이터에 대한 색인 리스트 생성
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
# print(text_processed[:4])       # 개개의 입력 문장이 [[ 44 455   0 809 703 667  62   9   0  87 120 366   0 152   0   0  66  56 0 136   0   0   0   0   0], ...]

# 학습 데이터를 적절히 섞자
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))    # 0부터 적힌 크기 만큼 숫자로 구성된 리스트를 만들고, 순서를 섞는다
x_shuttled = text_processed[shuffled_ix]    # suffled_ix에 적힌 순서대로 넘파이 리스트 내의 원소들의 순서를 바꾼다
y_shuffled = text_data_target[shuffled_ix]
print(y_shuffled[:4])

# 학습 데이터를 80:20 비율로 학습 및 테스트 데이터셋을 만든다
ix_cutoff = int(len(y_shuffled)*0.80)
x_train, x_test = x_shuttled[:ix_cutoff], x_shuttled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d}--{:d}".format(len(y_train), len(y_test)))

# 그래프 플레이스홀더 선언
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# 임베딩 행렬을 생성하고 입력 데이터 x에 대한 임베딩 값을 조회하는 연산 생성
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))     # 문장(단어) --> 문장(인덱스) --> embedding vector(50dim)
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)        # x_data에 대한 임베딩 값을 조회하는 연산 생성

# 모델 선언
cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)     # 입력데이터 길이를 고정했지만, 가변RNN이 속도가 빨라서 쓴다
output = tf.nn.dropout(output, dropout_keep_prob)

# RNN 결과를 재배열하고 마지막 출력값을 자른다
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)    # 마지막 output 값 버리기

# RNN결과를 FC층으로 연결 후 소프트맥스
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

# 비용 함수 선언. sparse_softmax 사용시, 정수 색인 값과 실수형 로짓 값 필요
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32  label=int32
loss = tf.reduce_mean(losses)

# 성능 테스트 함수 선언
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

# 최적화 함수 선언
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#########################################################
# 학습 하기
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
for epoch in range(epochs):
    # 학습데이터 섞기
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train) / batch_size) + 1
    for i in range(num_batches):
        # 학습 데이터 선택
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        # 학습실행
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
    # 배치에 대한 비용, 정확도 계산
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)

    # Evaluation 실행
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    # Epoch: 20, Test Loss: 0.46, Test Acc: 0.86

# 실험결과를 그림으로 그리면...
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

