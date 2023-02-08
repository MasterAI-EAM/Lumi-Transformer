import cv2
import numpy as np
import os

pic_path = './result/'
pic_target = '/Users/yifanguang/Programming/CodeProjects/CV/picture/'

num_width_list = []
num_lenght_list = []

picture_names =  os.listdir(pic_path)
if len(picture_names)==0:
    print("no such file")
 
else:
    
    img_1_1 = cv2.imread(pic_path + '1_1.jpg')
    (width, length, depth) = img_1_1.shape

    for picture_name in picture_names:
        num_width_list.append(int(picture_name.split("_")[0]))
        num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))

    num_width = max(num_width_list)
    num_length = max(num_lenght_list)

    splicing_pic = np.zeros((num_width*width, num_length*length, depth))

    for i in range(1, num_width+1):
        for j in range(1, num_length+1):
            img_part = cv2.imread(pic_path + '{}_{}.jpg'.format(i, j))
            splicing_pic[width*(i-1) : width*i, length*(j-1) : length*j, :] = img_part

    cv2.imwrite(pic_target + 'result.jpg', splicing_pic)
 