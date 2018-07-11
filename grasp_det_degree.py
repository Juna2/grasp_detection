#!/usr/bin/env python
''' 
Training a network on cornell grasping dataset for detecting grasping positions.
'''
import sys
import argparse
import os.path
import glob
import cv2
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
import grasp_img_proc
from grasp_inf import inference
import time

'''This script can train and test image data by using option "--train_or_validation" but the test results are printed in terminal'''

TRAIN_FILE = '/home/irobot2/Documents/custom_dataset/train-cgd'  # '/root/dataset/cornell_grasping_dataset/train-cgd'
VALIDATE_FILE = '/home/irobot2/Documents/custom_dataset/validation-cgd'  # '/root/dataset/cornell_grasping_dataset/validation-cgd'
MODEL_SAVE_PATH = '/home/irobot2/Documents/grasp_detection/models/grasp'

'''
1. Check TRAIN_FILE, VALIDATE_FILE, MODEL_SAVE_PATH
2. Check right starting Model(the model you want to continue to train) is in right place 
(see this line "default=MODEL_SAVE_PATH + '/m4_origin/m4.ckpt',")
3. Check "step = 8000" line if step variable is right
3. You also should check if there are any empty .txt file
'''


# def bboxes_to_grasps(bboxes):
#     # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
#     box = tf.unstack(bboxes, axis=1)
#     x = (box[0] + (box[4] - box[0])/2)  # x = (box[0] + (box[4] - box[0])/2) * 0.35
#     y = (box[1] + (box[5] - box[1])/2)  # y = (box[1] + (box[5] - box[1])/2) * 0.47
#     tan = (box[3] -box[1]) / (box[2] -box[0])  # tan = (box[3] -box[1]) / (box[2] -box[0]) *0.47/0.35 
#     h = tf.sqrt(tf.pow((box[2] -box[0]), 2) + tf.pow((box[3] -box[1]), 2)) # h = tf.sqrt(tf.pow((box[2] -box[0])*0.35, 2) + tf.pow((box[3] -box[1])*0.47, 2))
#     w = tf.sqrt(tf.pow((box[6] -box[0]), 2) + tf.pow((box[7] -box[1]), 2)) # w = tf.sqrt(tf.pow((box[6] -box[0])*0.35, 2) + tf.pow((box[7] -box[1])*0.47, 2))
#     return x, y, tan, h, w

def bboxes_to_grasps(bboxes):
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    box = tf.unstack(bboxes, axis=1)
    x = (box[0] + (box[4] - box[0])/2)  # x = (box[0] + (box[4] - box[0])/2) * 0.35
    y = (box[1] + (box[5] - box[1])/2)  # y = (box[1] + (box[5] - box[1])/2) * 0.47
    tan = (box[3] -box[1]) / (box[2] -box[0])  # tan = (box[3] -box[1]) / (box[2] -box[0]) *0.47/0.35
    tan_confined = tf.minimum(11., tf.maximum(-11., tan)) 
    degree = tf.atan(tan)*180/3.14
    h = tf.sqrt(tf.pow((box[2] -box[0]), 2) + tf.pow((box[3] -box[1]), 2)) # h = tf.sqrt(tf.pow((box[2] -box[0])*0.35, 2) + tf.pow((box[3] -box[1])*0.47, 2))
    w = tf.sqrt(tf.pow((box[6] -box[0]), 2) + tf.pow((box[7] -box[1]), 2)) # w = tf.sqrt(tf.pow((box[6] -box[0])*0.35, 2) + tf.pow((box[7] -box[1])*0.47, 2))
    return x, y, degree, h, w

def grasp_to_bbox(x, y, tan, h, w):
    theta = tf.atan(tan)
    edge1_x = x -w/2*tf.cos(theta) +h/2*tf.sin(theta)
    edge1_y = y -w/2*tf.sin(theta) -h/2*tf.cos(theta)
    edge2_x = x +w/2*tf.cos(theta) +h/2*tf.sin(theta)
    edge2_y = y +w/2*tf.sin(theta) -h/2*tf.cos(theta)
    edge3_x = x +w/2*tf.cos(theta) -h/2*tf.sin(theta)
    edge3_y = y +w/2*tf.sin(theta) +h/2*tf.cos(theta)
    edge4_x = x -w/2*tf.cos(theta) -h/2*tf.sin(theta)
    edge4_y = y -w/2*tf.sin(theta) +h/2*tf.cos(theta)

    edge1 = (edge1_x, edge1_y)
    edge2 = (edge2_x, edge2_y)
    edge3 = (edge3_x, edge3_y)
    edge4 = (edge4_x, edge4_y)
    
    return [edge1, edge2, edge3, edge4]

def run_training():
    print(FLAGS.train_or_validation)
    if FLAGS.train_or_validation == 'train':
        print('distorted_inputs')
        data_files_ = TRAIN_FILE
        images, bboxes = grasp_img_proc.distorted_inputs(
                  [data_files_], FLAGS.num_epochs, batch_size=FLAGS.batch_size)
    else:
        print('inputs')
        data_files_ = VALIDATE_FILE
        images, bboxes = grasp_img_proc.inputs([data_files_])
    
    x, y, degree, h, w = bboxes_to_grasps(bboxes) # x, y, tan, h, w = bboxes_to_grasps(bboxes)
    
    # images_np = np.array(images)
    x_hat, y_hat, degree_hat, h_hat, w_hat = tf.unstack(inference(images), axis=1) # list
    # x_hat, y_hat, tan_hat, h_hat, w_hat = tf.unstack(inference(images), axis=1) # list
    # tangent of 85 degree is 11 

    # tan_hat_confined = tf.minimum(11., tf.maximum(-11., tan_hat))
    # tan_confined = tf.minimum(11., tf.maximum(-11., tan))
    # Loss function
    gamma = tf.constant(0.001)
    loss = tf.reduce_sum(tf.pow(x_hat -x, 2) +tf.pow(y_hat -y, 2) + gamma*tf.pow(degree_hat - degree/2, 2) +tf.pow(h_hat -h, 2) +tf.pow(w_hat -w, 2))
    train_op = tf.train.AdamOptimizer(epsilon=0.1).minimize(loss)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #save/restore model
    d={}
    l = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2']
    for i in l:
        d[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
    
    dg={}
    lg = ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4', 'w5', 'b5', 'w_fc1', 'b_fc1', 'w_fc2', 'b_fc2', 'w_output', 'b_output']
    for i in lg:
        dg[i] = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == i+':0'][0]
    
    saver = tf.train.Saver(d)
    saver_g = tf.train.Saver(dg, max_to_keep=20)
    #saver.restore(sess, "/root/grasp/grasp-detection/models/imagenet/m2/m2.ckpt")
    saver_g.restore(sess, FLAGS.model_path)
    try:
        count = 0
        step = 0
        start_time = time.time()
        while not coord.should_stop():
            start_batch = time.time()
            #train  
            if FLAGS.train_or_validation == 'train':
                _, loss_value, images_np, x_value, x_model, y_value, y_model, degree_value, degree_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, images, x, x_hat, y, y_hat, degree, degree_hat, h, h_hat, w, w_hat])
                # _, loss_value, images_np, x_value, x_model, y_value, y_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([train_op, loss, images, x, x_hat, y, y_hat, tan, tan_hat, h, h_hat, w, w_hat])
                duration = time.time() - start_batch
                if step % 100 == 0:
                    degree_model_x2 = [a*2 for a in degree_model[:3]]
                    print('Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | y = %s\n | y_hat = %s\n | degree = %s\n | degree_hat = %s\n | degree_hat - degree = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n') \
                        %(step, loss_value, x_value[:3], x_model[:3], y_value[:3], y_model[:3], degree_value[:3], degree_model_x2, degree_model_x2 - degree_value[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration)
                    # print('Step %d | loss = %s\n | x = %s\n | x_hat = %s\n | y = %s\n | y_hat = %s\n | tan = %s\n | tan_hat = %s\n | h = %s\n | h_hat = %s\n | w = %s\n | w_hat = %s\n | (%.3f sec/batch\n')%(step, loss_value, x_value[:3], x_model[:3], y_value[:3], y_model[:3], tan_value[:3], tan_model[:3], h_value[:3], h_model[:3], w_value[:3], w_model[:3], duration)
                    cv2.imshow('bbox', images_np[1]) ###################################
                    # print(images_np[1])
                    # cv2.imwrite("./"+str(step)+".png", images_np[1]);
                    cv2.waitKey(1)
                if step % 1000 == 0:
                    filename = MODEL_SAVE_PATH + '/step_' + str(step)
                    if not os.path.exists(filename):
                        os.mkdir(filename)             
                    saver_g.save(sess, filename + '/m4.ckpt')
                    
                    # cv2.imshow('bbox', images_np[1]) ###################################
                    # cv2.waitKey(0)
                    # print(images_np[0])
                    
                    # if step == 100000:
                    #     sess.close()
                    
            # else:
            #     bbox_hat = grasp_to_bbox(x_hat, y_hat, tan_hat, h_hat, w_hat)
            #     bbox_value, bbox_model, x_value, x_model, y_value, y_model, tan_value, tan_model, h_value, h_model, w_value, w_model = sess.run([bboxes, bbox_hat, x, x_hat, y, y_hat, tan, tan_hat, h, h_hat, w, w_hat])
            #     bbox_value = np.reshape(bbox_value, -1)
            #     bbox_value = [(bbox_value[0],bbox_value[1]),(bbox_value[2],bbox_value[3]),(bbox_value[4],bbox_value[5]),(bbox_value[6],bbox_value[7])]  # bbox_value = [(bbox_value[0]*0.35,bbox_value[1]*0.47),(bbox_value[2]*0.35,bbox_value[3]*0.47),(bbox_value[4]*0.35,bbox_value[5]*0.47),(bbox_value[6]*0.35,bbox_value[7]*0.47)]
            #     p1 = Polygon(bbox_value)
            #     p2 = Polygon(bbox_model)
            #     iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area) 
            #     angle_diff = np.abs(np.arctan(tan_model)*180/np.pi -np.arctan(tan_value)*180/np.pi)
            #     duration = time.time() -start_batch
            #     # if angle_diff < 30. and iou >= 0.25:
            #     count+=1
            #     print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
            #     print('x=',x_value,x_model,' y=',y_value, y_model,' tan=',tan_value, tan_model,' h=', h_value, h_model,' w=', w_value, w_model)
            step +=1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps, %.1f min.' % (FLAGS.num_epochs, step, (time.time()-start_time)/60))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def main(_):
    run_training()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/root/imagenet-data',
        help='Directory with training data.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tf',
        help='Tensorboard log_dir.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=MODEL_SAVE_PATH + '/m4_origin/m4.ckpt',
        help='Variables for the model.'
    )
    parser.add_argument(
        '--train_or_validation',
        type=str,
        default='train',
        help='Train or evaluate the dataset'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

'''./grasp_det.py --model_path /home/irobot2/Documents/grasp_detection/models/grasp/m4/m4.ckpt --train_or_validation validation'''
'''--num_epochs 5'''
