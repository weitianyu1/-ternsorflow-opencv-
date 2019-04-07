
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import dlib
import sys

detector = dlib.get_frontal_face_detector() #获取人脸分类器

ID=(1001,1002)

cascade_path = "E:\\cv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"

color = (0, 255, 0)

w=128
h=128
c=3

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def CNNlayer():
    #第一个卷积层（128——>64)
    conv1=tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #第二个卷积层(64->32)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #第三个卷积层(32->16)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    #第四个卷积层(16->8)
    conv4=tf.layers.conv2d(
          inputs=pool3,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])

    #全连接层
    dense1 = tf.layers.dense(inputs=re1,
                          units=1024,
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1,
                          units=512,
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits= tf.layers.dense(inputs=dense2,
                            units=2,
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits
#---------------------------网络结束---------------------------
logits=CNNlayer()
predict = tf.argmax(logits, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'ckpt1/faces.ckpt-4')


user=input("图片（G）还是摄像头（V）:")
if user=="G":
    path=input("图片路径名是：")
    img = cv2.imread(path)
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for index, face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        io.imsave('temp.png', img)
        img1=io.imread('temp.png')
        img1=transform.resize(img1,(w,h,c))
        cv2.imshow('image',img1)

        img1 = img[top:bottom,left:right]
        img1=transform.resize(img1,(w,h,c))
        # cv2.imshow('image1',img)
        res=sess.run(predict, feed_dict={x:[img1]})
        print(ID[res[0]])
    if len(dets)==0:
        img=transform.resize(img,(w,h,c))
        res=sess.run(predict, feed_dict={x:[img]})
        print(ID[res[0]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 视屏封装格式

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cascade = cv2.CascadeClassifier(cascade_path)
        # faceRects = cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        # if len(faceRects) > 0:
        #     for faceRect in faceRects:
        #         x, y, w, h = faceRect
        #         cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
        cv2.imshow('frame', frame)
        # 抓取图像
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('image/now.png', frame)

            img = cv2.imread("image/now.png")
            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for index, face in enumerate(dets):
                print('face {}; left {}; top {}; right {}; bottom {}'.format(index,
                    face.left(), face.top(), face.right(), face.bottom()))
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                img = img[top:bottom,left:right]


            #img=io.imread('image/now.png')
            img=transform.resize(img,(w,h,c))
            res=sess.run(predict, feed_dict={x:[img]})
            print(ID[res[0]])
        k=cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()



























