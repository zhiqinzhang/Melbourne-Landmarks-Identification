"""
This file defines some methods of image batch processing.
Author: Zhiqin Zhang
Update Date: 08/04/2019
use conda deactivate
"""
import cv2 as cv
import os
import settings as st
import random
import shutil
import tensorflow as tf

def img_resize(src_path, dst_path):
    for filename in os.listdir(src_path):
        print(filename)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        src_file = os.path.join(src_path,filename)
        dst_file = os.path.join(dst_path,filename)
        if ".DS_Store" not in src_file:
            if os.path.isfile(src_file):
                if not os.path.isfile(dst_file):
                    print(src_file)
                    img = cv.imread(src_file)
                    res_img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
                    cv.imwrite(dst_file, res_img)
            else:
                img_resize(src_file, dst_file)


def move_random_n_percent_file(src_dir, dst_dir, n):
    num = int(len(os.listdir(src_dir))*(n/100))
    print(num)
    samples = random.sample(os.listdir(src_dir),num)
    for sample in samples:
        # shutil.copyfile(os.path.join(src_dir,sample),
        #                 os.path.join(dst_dir,sample))
        shutil.move(os.path.join(src_dir, sample),
                    os.path.join(dst_dir, sample))


def select_n_percent_imgs_per_label(n, dst_path=os.path.join(st.ROOT,'test_data'),
                                    src_path=os.path.join(st.ROOT,'data'),):
    for dirname in os.listdir(src_path):
        print(dirname)
        dir_path = os.path.join(dst_path,dirname)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        src_dir = os.path.join(src_path,dirname)
        dst_dir = os.path.join(dst_path,dirname)
        if not os.path.isfile(src_dir):
            move_random_n_percent_file(src_dir, dst_dir, n)


def clear_old_test_data(dst_path=os.path.join(st.ROOT,'test_data')):
    for dirname in os.listdir(dst_path):
        dir_path = os.path.join(dst_path,dirname)
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)



def get_input_and_output_node(output_graph_path):
    graph = tf.GraphDef()
    graph.ParseFromString(open(output_graph_path,'rb').read())
    for n in graph.node:
        print(n.name + "-->" +n.op)



if __name__=='__main__':
    img_resize(os.path.join(st.ROOT,'data'),os.path.join(st.ROOT,'res_data'))
    img_resize(os.path.join(st.ROOT,'test_data'),os.path.join(st.ROOT,'res_test_data'))
    # clear_old_test_data()
    # select_n_percent_imgs_per_label(10)
    # clear_old_test_data()
    # test = os.path.join(st.MOBILENET_ROOT,'output')
    # test = os.path.join(test,'mobilenet_v1_100_224_quant_output_graph.pb')
    # convert_saved_model_to_tflite(test)
    # get_input_and_output_node(test)
    # convert_saved_model_to_mlmodel(test,os.path.join(st.OUTPUT_ROOT,'test.mlmodel'))
