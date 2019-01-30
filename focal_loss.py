# -*- coding: utf-8 -*-
import tensorflow as tf

"""
Tensorflow实现何凯明的Focal Loss, 该损失函数主要用于解决分类问题中的类别不平衡
focal_loss_sigmoid: 二分类loss
focal_loss_softmax: 多分类loss
Reference Paper : Focal Loss for Dense Object Detection
"""

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)**gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L

if __name__ == '__main__':
    logits=tf.random_uniform(shape=[5],minval=-1,maxval=1,dtype=tf.float32)
    labels=tf.Variable([0,1,0,0,1])
    loss1=focal_loss_sigmoid(labels=labels,logits=logits)

    logits2=tf.random_uniform(shape=[5,4],minval=-1,maxval=1,dtype=tf.float32)
    labels2=tf.Variable([1,0,2,3,1])
    loss2=focal_loss_softmax(labels==labels2,logits=logits2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(loss1)
        print sess.run(loss2)

