# -*- coding: utf-8 -*-
"""
作者：Shayne
程式簡介：練習用的CNN範例
"""

import tensorflow as tf

def read_and_decode(filename, BATCH_SIZE, MAX_EPOCH): 
    # 建立文件名隊列
    filename_queue = tf.train.string_input_producer([filename], 
                                                    num_epochs=MAX_EPOCH)
    
    # 數據讀取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # 數據解析
    img_features = tf.parse_single_example(
            serialized_example,
            features={ 'Label'    : tf.FixedLenFeature([], tf.int64),
                       'image_raw': tf.FixedLenFeature([], tf.string), })
    
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [42, 42])
    
    label = tf.cast(img_features['Label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch =tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size = BATCH_SIZE,
                                 capacity = 1000 + 3 * BATCH_SIZE,
                                 min_after_dequeue = 1000)

    return image_batch, label_batch

def Weight(shape, mean=0, stddev=1):
    init = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(init)

def bias(shape, mean=0, stddev=1):
    init = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(init)

def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')

# ========================================================================#
#                              主程式區                                   #
# ========================================================================#
    
filename = './py_Train.tfrecords'
BATCH_SIZE = 128
MAX_EPOCH = 20
LABEL_NUM = 10

# Input images & labels
X  = tf.placeholder(tf.float32, shape = [None, 42, 42, 1])
y_ = tf.placeholder(tf.float32, shape = [None, LABEL_NUM])

# 卷積網路建立
# Conv1
W_conv1 = Weight([5, 5, 1, 16])
b_conv1 = bias([16])
y_conv1 = conv2d(X, W_conv1) + b_conv1
# ReLU1
relu1 = tf.nn.relu(y_conv1)
# Pool1
pool1 = max_pool(relu1)
# Conv2
W_conv2 = Weight([3, 3, 16, 32])
b_conv2 = bias([32])
y_conv2 = conv2d(pool1, W_conv2) + b_conv2
# ReLU2
relu2 = tf.nn.relu(y_conv2)
# Pool2
pool2 = max_pool(relu2)
# FC1
W_fc1 = Weight([11*11*32, 64])
b_fc1 = bias([64])
# 要把前面的網路的輸出壓扁成一維陣列
h_flat = tf.reshape(pool2, [-1, 11*11*32])
y_fc1 = tf.matmul(h_flat, W_fc1) + b_fc1
# ReLU3
relu3 = tf.nn.relu(y_fc1)
# FC2 - 輸出層
W_fc2 = Weight([64, 10]) # 分類0-9，共 10 類
b_fc2 = bias([10])

# 訓練用
y = tf.matmul(relu3, W_fc2) + b_fc2

# 預測用
# 預測時，會先經過 softmax 計算，再取最大值為預測結果
y_pred = tf.argmax(tf.nn.softmax(y), 1)

# Cost optimizer
lossFcn = tf.nn.softmax_cross_entropy_with_logits_v2
cost = tf.reduce_mean(lossFcn(labels=y_, logits=y))

# 使用 AdamOptimizer 進行迭代
train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

# 計算正確率
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 設定輸入資料來源
img_bat, lb_bat = read_and_decode(filename, BATCH_SIZE, MAX_EPOCH)
train_x = tf.reshape(img_bat, [-1, 42, 42, 1])
train_y = tf.one_hot(lb_bat, LABEL_NUM)

with tf.Session() as sess:
    # 變數初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # 啟動 TFRecord 佇列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    i = 0
    try:
        while not coord.should_stop():
            image_train, label_train = sess.run([train_x, train_y])
            sess.run(train_step, feed_dict={ X : image_train, 
                                             y_: label_train})
            
            if i % 50 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={ X  : image_train, 
                                                                y_ : label_train})

                print('Iter %d, accuracy %4.2f%%' % (i,train_accuracy*100))
            i += 1
            
    except tf.errors.OutOfRangeError:
        print('Done!')
        
    finally:
        coord.request_stop()
            
    coord.join(threads)