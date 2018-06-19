# coding = utf-8
"""
    Created on 2017 10.17
    @author: liupeng
    wechat: lp9628
    blog: http://blog.csdn.net/u014365862/article/details/78422372
"""
from deeplab_model import deeplab_v3_plus_generator
import preprocessing

import tensorflow as tf 
from tensorflow.python.framework import graph_util
import cv2
import numpy as np
import os
import sys
from utils import preprocessing

_R_MEAN = 127.59
_G_MEAN = 122.33
_B_MEAN = 116.00

MODEL_DIR = "model_hair/"
MODEL_NAME = "frozen_hair_model.pb"
if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

X = tf.placeholder(tf.float32, [None,None, None, 3],name = "inputs_placeholder1")
network = deeplab_v3_plus_generator(2,16,'resnet_v2_101','./resnet_v2_101/resnet_v2_101.ckpt',0.9997)
logits = network(X, False)  # logits维度[n,h,w,2]
predict = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3,name = 'predictions')

def freeze_graph(model_folder):
    #checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    #input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    input_checkpoint = model_folder
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    output_node_names = "predictions" #原模型输出操作节点的名字
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.
    saver = tf.train.Saver()

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        #print "predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name:",op.name)
        print ("success!")

        #下面是用于测试， 读取pd模型，答应每个变量的名字。
        graph = load_graph("model_supervisely/frozen_hair_model.pb")
        for op in graph.get_operations():
            #print(op.name, op.values())
            print("name111111111111:",op.name)
        pred = graph.get_tensor_by_name('prefix/inputs_placeholder1:0')
        print (pred)
        temp = graph.get_tensor_by_name('prefix/predictions:0')
        print (temp)

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

if __name__ == '__main__':
    # 1.验证ckpt的有效性
    model_folder = './model_supervisely/model.ckpt-68553'
    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, model_folder)
    """
    image = cv2.imread('person1.jpg')
    image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    print (image.shape)
    r, g, b = cv2.split(image)
    r = r - _R_MEAN
    g = g - _G_MEAN
    b = b - _B_MEAN
    image = cv2.merge([r,g,b])
    #image[0] = image[:,:,0] - _R_MEAN
    #image[1] = image[:,:,1] - _G_MEAN
    #image[2] = image[:,:,2] - _B_MEAN
    #h, w, c = image.shape
    """
    
    image_string = tf.read_file('person1.jpg')
    image = tf.image.decode_image(image_string) #解码jpeg后是uint8类型的
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8)) #转为float32类型的，但并不进行归一化
    image.set_shape([None, None, 3])
    image = preprocessing.mean_image_subtraction(image)
    image = sess.run(image)
    
    mask = sess.run(predict,feed_dict= {X:[image]})
    mask = (mask[0] == 1)*255
    cv2.imwrite('person1_mask_src.jpg', mask)

    #print (mask)

    # 2. 固化
    #model_folder = './model_hair/model.ckpt-20905'
    #freeze_graph(model_folder)
