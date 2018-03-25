import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

def conv2d(name, X, W, b, strides = 1):
    X = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding="SAME")
    X = tf.nn.bias_add(X, b)
    return tf.nn.relu(X, name=name)

def maxpooling(name,X, k=2):
    return tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME', name=name)

#定义网络
weights = {
    "wc1": tf.Variable(tf.random_normal([5, 5, 1, 6])),
    "wc2": tf.Variable(tf.random_normal([5, 5, 6, 16])),
    "wd1": tf.Variable(tf.random_normal([7*7*16, 120])),
    "wd2": tf.Variable(tf.random_normal([120, 84])),
    "out": tf.Variable(tf.random_normal([84, 10]))
}

biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[6])),
    "bc2": tf.Variable(tf.constant(0.0, shape=[16])),
    "bd1": tf.Variable(tf.constant(0.0, shape=[120])),
    "bd2": tf.Variable(tf.constant(0.0, shape=[84])),
    "out": tf.Variable(tf.constant(0.0, shape=[10]))
}

#
def lenet_inference(x, weights, biases, dropout):
    #reshape
    x = tf.reshape(x,shape=[-1, 28, 28, 1])
    #第一层conv1层 6@28*28
    conv1 = conv2d("layer1_conv", x, weights["wc1"], biases["bc1"])
    pool1 = maxpooling("pool1", conv1, k=2)

    #第二层conv2  16@10*10
    conv2 = conv2d("layer2_conv", pool1, weights["wc2"], biases["bc2"])
    pool2 = maxpooling("pool2", conv2, k=2)
    print(pool2.shape)

    #第三层FC1 全连接
    fc1 = tf.reshape(pool2, shape=[-1, weights["wd1"].get_shape().as_list()[0]])
    print(weights["wd1"], biases["bc1"], weights["wd1"].get_shape().as_list()[0])
    fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"])
    fc1 = tf.nn.relu(fc1)

    #第四层全连接层
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights["wd2"]), biases["bd2"])
    fc2 = tf.nn.relu(fc2)
    # # dropout
    # fc2 = tf.nn.dropout(fc2, 0.75)
    #第五层输出层
    out = tf.add(tf.matmul(fc2, weights["out"]), biases["out"])
    return out

#
dropout = 0.5
#学习速率
leaning_rate = 0.01
#迭代次数
iter_times = 20000
#批处理样本数
batch_sizes = 128
x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

def main(argv=None):
    #构建模型
    pred = lenet_inference(x, weights, biases, dropout)

    #定义损失和优化器
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
    optimizer = tf.train.AdamOptimizer(leaning_rate).minimize(cost)

    #评估正确率
    corrent_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(corrent_pred, dtype=tf.float32))
    global_step = tf.Variable(0, name="global_step", trainable=False)

    #初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    start = global_step.eval()  # 得到 global_step 的初始值
    step = 1
    #
    while step * batch_sizes < iter_times:
        batch_x, batch_y = mnist.train.next_batch(batch_sizes)
        #train
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #cost, acc
        if step % 500 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y:batch_y})
            print("step: " + str(step)+"      loss: {:.6f}".format(loss) + "      acc: {:.5f}".format(acc))
        step += 1
        global_step.assign(step).eval()  #更新计数器

    loss, acc = sess.run([cost, accuracy], feed_dict={x: mnist.test.images[:256], y:mnist.test.labels[:256]})
    print("Test------ step: " + str(step)+"      loss: {:.6f}".format(loss) + "      acc: {:.5f}".format(acc))


if __name__ == "__main__":
    #tf.app.run()
    main()

