# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:28:40 2019

@author: mingliangbai
"""
#
##load library

#dropout L1 L2 BN  batch  learning rate  adam /#sgd with moment  hidden 

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import GRU#from keras.layers.recurrent import SimpleRNN  #from keras.layers.recurrent import LSTM
from keras import layers
from keras.optimizers import Adam
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

##########################
#load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./minist', one_hot=True)  # they has been normalized to range (0,1)
X_test= mnist.test.images#[:2000]
Y_test= mnist.test.labels#[:2000]
X_train = mnist.train.images
Y_train = mnist.train.labels
X_train = X_train.reshape(-1,28,28)
X_test = X_test.reshape(-1,28,28)
n_classes = 10

###################################
# create and fit the SimpleRNN model
model = Sequential()
model.add(GRU(units=16, activation='relu', input_shape=(28,28)))
#model.add(GRU(units=8, activation='relu', dropout=0.1, recurrent_dropout=0.5))
#model.add(SimpleRNN(units=16, activation='relu', input_shape=(28,28)))#SimpleRNN
#model.add(LSTM(units=16, activation='relu', input_shape=(28,28)))#LSTM
#model.add(GRU(units=16, activation='relu', input_shape=(784,1)))
#如何添加两个GRU layer???
model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train,
                    batch_size=100,
                    epochs=2)
y_pre=model.predict(X_test)
y_predict=np.argmax(y_pre,axis=1)#predicted label of test data
score = model.evaluate(X_test, Y_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])


######################################################
# 保存模型
model.save('model.h5')   # HDF5文件，pip install h5py
# 保存参数，载入参数
model.save_weights('my_model_weights.h5')
#https://blog.csdn.net/jclian91/article/details/83038861读取权重


##############################################################
#print the weight and bias of model.h5
import h5py






####################################################
# 载入网络结构不经过微调，直接预测测试集的输出
model1 = keras.models.load_model("model.h5")
model1.summary()
y_pre=model1.predict(X_test)
y_predict=np.argmax(y_pre,axis=1)#predicted label of test data
score = model1.evaluate(X_test, Y_test)
#经过测试，测试集的分类精度与model.h5的分类精度相同，证明之前训练的模型被成功加载了



#########################################################
# 载入网络结构，微调最后一层，并预测测试集的输出
model2 = keras.models.load_model("model.h5")
model2.layers[0].trainable=False#保留原权重，不进行微调
model2.layers[1].trainable=True#进行微调的层
model2.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model2.summary()
model2.fit(X_train, Y_train,
                    batch_size=100,
                    epochs=10)
y_pre2=model2.predict(X_test)
y_predict2=np.argmax(y_pre,axis=1)#predicted label of test data
score2 = model2.evaluate(X_test, Y_test)
#经过测试，测试集的分类精度为0.8949，这比model.h5的分类精度0.8859高，证明确实进行了微调，微调是有效果的




'''uncorrected references
#model.load_weights('my_model_weights.h5')#https://blog.csdn.net/jiandanjinxin/article/details/77152530
##需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，可以通过层名字来加载模型  
##model.load_weights('my_model_weights.h5', by_name=True)
##from keras.models import model_from_json
##json_string = model.to_json()
##model1 = model_from_json(json_string)
#
#model1 = Sequential()
#model1.add(GRU(units=16, activation='relu', input_shape=(28,28)))
#model1.add(Dense(n_classes))
#model1.add(Activation('softmax'))
#
##https://blog.csdn.net/Tourior/article/details/83822944设置网络前两层不可训练，使用预训练模型的权重
#model1.layers[0].trainable=False
#model1.layers[1].trainable=False
#model.load_weights('my_model_weights.h5')
#model1.compile(loss='categorical_crossentropy',
#              optimizer=Adam(lr=0.001),
#              metrics=['accuracy'])
#model1.fit(X_train, Y_train,
#                    batch_size=100,
#                    epochs=2)
#y_pre=model1.predict(X_test)
#y_predict=np.argmax(y_pre,axis=1)#predicted label of test data
#score = model1.evaluate(X_test, Y_test)
##--------------------- 
##作者：帅气的弟八哥 
##来源：CSDN 
##原文：https://blog.csdn.net/jiandanjinxin/article/details/77152530 
##版权声明：本文为博主原创文章，转载请附上博文链接！
##--------------------- 
##作者：Young_618 
##来源：CSDN 
##原文：https://blog.csdn.net/cymy001/article/details/78647640 
##版权声明：本文为博主原创文章，转载请附上博文链接！'''

