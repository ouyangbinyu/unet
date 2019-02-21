import os
import tensorflow as tf
import numpy as np
import unet_inference
import scipy.io as sio
from unet_inference import IMAGE_SIZE,BATCH_SIZE,NUMS_CHANNELS,IMAGE_NUM
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DACAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH ="C:\\Users\\xmu\\PycharmProjects\\unet"
MODEL_NAME = "my_model"



Training_data_inputs_Name ='train100_256_256_1.mat'
Training_inputs = sio.loadmat(Training_data_inputs_Name)
Training_inputs_data = Training_inputs['image_train']

Training_inputs_data = np.expand_dims(Training_inputs_data,3)

Training_data_labels_Name = 'label100_256_256_1.mat'
Training_labels = sio.loadmat(Training_data_labels_Name)

Training_labels_data = Training_labels['b']
Training_labels_data = np.expand_dims(Training_labels_data,3)


def train():
    x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUMS_CHANNELS],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUMS_CHANNELS],name = 'y-input')
    #regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = unet_inference.inference(x)
    global_step = tf.Variable(0,trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    base_loss =tf.reduce_mean(tf.square(y - y_))
    loss = base_loss #+ tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,IMAGE_NUM/BATCH_SIZE,LEARNING_RATE_DACAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name = 'train')
    #初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            start = (i*BATCH_SIZE)%100
            end = min(start+BATCH_SIZE,100)
            xs = Training_inputs_data[start:end,:,:,:]
            ys = Training_labels_data[start:end,:,:,:]
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            print(i)
            if i%1000 == 0:
                print("after %d training step(s),loss on training batch is %g."%(step,loss_value))
                #保存当前的模型
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)

def main(argv=None):
    train()
if __name__ == '__main__':
    tf.app.run()

