import tensorflow as tf

#我做的修改  git
INPUT_NODE = 256*256
OUTPUT_NODE = 256*256

IMAGE_SIZE = 256
NUMS_CHANNELS = 1
IMAGE_NUM = 64
BATCH_SIZE = 10
#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 3

#第二层卷积层的尺寸和深度

CONV2_DEEP = 64
CONV2_SIZE = 3

#第四层卷积层的尺寸和深度

CONV3_DEEP = 128
CONV3_SIZE = 3

#第五层卷积层的尺寸和深度
CONV4_DEEP = 128
CONV4_SIZE = 3

#第七层卷积层的尺寸和深度

CONV5_DEEP = 256
CONV5_SIZE = 3

#第八层卷积层的尺寸和深度
CONV6_DEEP = 128
CONV6_SIZE = 3

#第十层卷积层的尺寸和深度
CONV7_DEEP = 128
CONV7_SIZE = 3

#第十一层卷积层的尺寸和深度
CONV8_DEEP = 64
CONV8_SIZE = 3

#第十三层卷积层的尺寸和深度
CONV9_DEEP = 64
CONV9_SIZE = 3

#第十四层卷积层的尺寸和深度
CONV10_DEEP = 64
CONV10_SIZE = 3

#第十五层卷积层的尺寸和深度
CONV11_DEEP = 1
CONV11_SIZE = 1


def inference(input_tensor):

    #第一层
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUMS_CHANNELS,CONV1_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    #第二层
    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    #第三层 池化
    with tf.variable_scope('layer3-pool1'):
        pool1 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
   # print(pool1.shape.as_list())
    #第四层
    with tf.variable_scope('layer4-conv3'):
        conv3_weights = tf.get_variable("weight",[CONV3_SIZE,CONV3_SIZE,CONV2_DEEP,CONV3_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias",[CONV3_DEEP],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool1,conv3_weights,strides=[1,1,1,1],padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
    #第五层
    with tf.variable_scope('layer5-conv4'):
        conv4_weights = tf.get_variable("weight",[CONV4_SIZE,CONV4_SIZE,CONV3_DEEP,CONV4_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias",[CONV4_DEEP],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3,conv4_weights,strides=[1,1,1,1],padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
    #print(relu4.shape.as_list())
    #第六层 池化
    with tf.variable_scope('layer6-pool2'):
        pool2 = tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #第七层
    with tf.variable_scope('layer7-conv5'):
        conv5_weights = tf.get_variable("weight",[CONV5_SIZE,CONV5_SIZE,CONV4_DEEP,CONV5_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias",[CONV5_DEEP],initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool2,conv5_weights,strides=[1,1,1,1],padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5,conv5_biases))
    #第八层
    with tf.variable_scope('layer8-conv6'):
        conv6_weights = tf.get_variable("weight",[CONV6_SIZE,CONV6_SIZE,CONV5_DEEP,CONV6_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias",[CONV6_DEEP],initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5,conv6_weights,strides=[1,1,1,1],padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6,conv6_biases))
    #第九层
    with tf.variable_scope('layer9-pool3'):
        tf.set_random_seed(1)
        kernel = tf.random_normal(shape=[2, 2, 128, 128])
        pool3 = tf.nn.conv2d_transpose(relu6,kernel,output_shape=[BATCH_SIZE,128,128,128],strides=[1,2,2,1],padding='SAME')
   # print(pool3.shape.as_list())
    #拼接操作
    concat1 = tf.concat([pool3,relu4],3)

    #第十层
    with tf.variable_scope('layer10-conv7'):
        conv7_weights = tf.get_variable("weight", [CONV7_SIZE, CONV7_SIZE, 256, CONV7_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv7_biases = tf.get_variable("bias", [CONV7_DEEP], initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(concat1, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, conv7_biases))

    #第十一层
    with tf.variable_scope('layer11-conv8'):
        conv8_weights = tf.get_variable("weight", [CONV8_SIZE, CONV8_SIZE, CONV7_DEEP, CONV8_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv8_biases = tf.get_variable("bias", [CONV8_DEEP], initializer=tf.constant_initializer(0.0))
        conv8 = tf.nn.conv2d(relu7, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu8 = tf.nn.relu(tf.nn.bias_add(conv8, conv8_biases))

    #第十二层
    with tf.variable_scope('layer12-pool4'):
        tf.set_random_seed(1)
        kernel2 = tf.random_normal(shape=[2, 2, 64, 64])
        pool4 = tf.nn.conv2d_transpose(relu8,kernel2,output_shape=[BATCH_SIZE,256,256,64],strides=[1,2,2,1],padding='SAME')

    concat2 = tf.concat([pool4, relu2],3)
    #print(concat2.shape.as_list())
    #第十三层
    with tf.variable_scope('layer13-conv9'):
        conv9_weights = tf.get_variable("weight", [CONV9_SIZE, CONV9_SIZE, 128, CONV9_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv9_biases = tf.get_variable("bias", [CONV9_DEEP], initializer=tf.constant_initializer(0.0))
        conv9 = tf.nn.conv2d(concat2, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu9 = tf.nn.relu(tf.nn.bias_add(conv9, conv9_biases))

    #第十四层
    with tf.variable_scope('layer14-conv10'):
        conv10_weights = tf.get_variable("weight", [CONV10_SIZE, CONV10_SIZE, CONV9_DEEP, CONV10_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv10_biases = tf.get_variable("bias", [CONV10_DEEP], initializer=tf.constant_initializer(0.0))
        conv10 = tf.nn.conv2d(relu9, conv10_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu10 = tf.nn.relu(tf.nn.bias_add(conv10, conv10_biases))
    with tf.variable_scope('layer15-con11'):
        conv11_weights = tf.get_variable("weight", [CONV11_SIZE, CONV11_SIZE, CONV10_DEEP, CONV11_DEEP],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv11_biases = tf.get_variable("bias", [CONV11_DEEP], initializer=tf.constant_initializer(0.0))
        conv11 = tf.nn.conv2d(relu10, conv11_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu11 = tf.nn.relu(tf.nn.bias_add(conv11, conv11_biases))

    return relu11
