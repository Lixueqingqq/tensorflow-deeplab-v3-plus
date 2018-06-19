import os
import sys
import numpy as np
import datetime
import tensorflow as tf
import cv2

filename = "person1.jpg"
pb_path = "model/frozen_supervisely_model.pb"
path_to_output1 = "person1_mask1_2.jpg"
path_to_output2 = "person1_mask2_2.jpg"

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

def mean_image_subtraction_my(image,RGB_means = [127.59,122.33,116.00]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print (image.shape)
    r, g, b = cv2.split(image)
    r = r - RGB_means[0]
    g = g - RGB_means[1]
    b = b - RGB_means[2]
    image = cv2.merge([r,g,b])
    return image

def mean_image_subtraction_src(image, means=(127.59,122.33,116.00)): #传入的image单张的是[h,w,c]
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image) # 按第2个维度切分，切分3份，每一份是一个列表
    for i in range(num_channels):  #3个列表(分别代表r/g/b通道)，每个列表减去相应的均值，python的广播
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels) # 将channels列表，串联成三通道

# 定义节点
graph = load_graph(pb_path)
X = graph.get_tensor_by_name('prefix/inputs_placeholder1:0')
predict = graph.get_tensor_by_name('prefix/predictions:0')                              # 输出纬度是[n,h,w,2]

# 预处理方式1
image1 = cv2.imread(filename)
image1 = mean_image_subtraction_my(image1)

# 预处理方式2
image_string = tf.read_file(filename)
image2 = tf.image.decode_image(image_string) #解码jpeg后是uint8类型的
image2 = tf.to_float(tf.image.convert_image_dtype(image2, dtype=tf.uint8)) #转为float32类型的，但并不进行归一化
image2.set_shape([None, None, 3])
image2 = mean_image_subtraction_src(image2)
sess = tf.Session()
image2 = sess.run(image2)
sess.close()

# 3.启动session
sess = tf.Session(graph=graph)
mask1 = sess.run(predict,feed_dict= {X:[image1]})
mask1 = (mask1[0] != 0)*255
#mask1 = (mask1[0] == 1)*255
mask2 = sess.run(predict,feed_dict= {X:[image2]})
mask2 = (mask2[0] != 0)*255
#mask2 = (mask2[0] == 1)*255

cv2.imwrite(path_to_output1, mask1)
cv2.imwrite(path_to_output2, mask2)
sess.close()
