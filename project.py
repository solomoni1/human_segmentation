import io
import os

import tensorflow as tf
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
    
if __name__ == '__main__':
    import argparse
    
    print(" Human segmentation start")
    parser = argparse.ArgumentParser()
    parser.add_argument("height", default="170", type=int, help="Insert your height in centimeter")
    args = parser.parse_args()
    height = args.height
    
    layers = []
    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_1_1'))
    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True))

    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_2_1'))
    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True))

    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64, name='conv_3_1'))
    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2'))
    layers.append(MaxPool2d(kernel_size=2, name='max_3'))

    network = convolutional_autoencoder.Network(layers)
    
    input_image = "test.jpg"
    output = "output"
    
    #body
    checkpoint = "model\_body"

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))

        segmentation = sess.run(network.segmentation_result, feed_dict={
            network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})

        th_body = np.array(
            [0 if x < 0.5 else 255 for x in segmentation[0].flatten()])
        th_body_m = np.reshape(th_body, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
        segmented_image = np.dot(segmentation[0], 255)

        cv2.imwrite(os.path.join(output, 'body.jpg'), segmented_image)
        cv2.imwrite(os.path.join(output, 'th_body.jpg'), th_body_m)

    #merging image
    Total_R= th_head+th_body+th_left_arm+th_right_leg
    Total_G= th_left_arm+th_body+th_right_arm
    Total_B= th_body+th_left_leg+th_right_leg

    for i in range (len(th_head)):
        if (Total_R[i]<10):
            Total_R[i]=0
        else:
            Total_R[i]=255
        if (Total_G[i]<10):
            Total_G[i]=0
        else:
            Total_G[i]=255
        if (Total_G[i]<10):
            Total_G[i]=0
        else:
            Total_G[i]=255
	
    result_R=np.reshape(Total_R, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
    result_G=np.reshape(Total_G, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
    result_B=np.reshape(Total_B, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH])
    result = np.dstack([result_B,result_G,result_R])
    
    cv2.imwrite(os.path.join(output, 'result.jpg'), result)
    cv2.imwrite('result.jpg', result)
    
    temp = Image.open('result.jpg')
    temp.show()
    
    print(" Human segmentation complete\n")
    print(" Measure start\n")
    
    measure(height, th_head_m, th_body_m, th_left_arm_m, th_right_arm_m, th_left_leg_m, th_right_leg_m)
