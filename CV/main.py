import cv2
import numpy as np
import os
import shutil
 
pic_path = 'CV/xixi.jpg'
pic_target = './result/' 

if not os.path.exists(pic_target):  
    os.makedirs(pic_target)
    
cut_width = 512
cut_length = 512

picture = cv2.imread(pic_path)

(width, length, depth) = picture.shape

pic = np.zeros((cut_width, cut_length, depth))


num_width = int(width / cut_width)
num_length = int(length / cut_length)

for i in range(0, num_width):
    for j in range(0, num_length):
        pic = picture[i*cut_width : (i+1)*cut_width, j*cut_length : (j+1)*cut_length, :]      
        result_path = pic_target + '{}_{}.jpg'.format(i+1, j+1)
        cv2.imwrite(result_path, pic)
