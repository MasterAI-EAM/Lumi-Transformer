# Copyright (C) 2018 Sergiu Deitsch
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from PIL import Image
import numpy as np
import os,shutil

DROP36 = True
SPLIT_RATIO = 0.05
mapper_dict = {0 : 0, 0.3333333333333333 : 1, 0.6666666666666666 : 2, 1.0 : 3} if not DROP36 else {0 : 0, 0.3333333333333333 : 9, 0.6666666666666666 : 9, 1.0 : 1}

def load_dataset(fname=None):
    if fname is None:
        # Assume we are in the utils folder and get the absolute path to the
        # parent directory.
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir))
        fname = os.path.join(fname, 'labels.csv')

    data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
                         'path', 'probability', 'type'])
    image_fnames = np.char.decode(data['path'])
    probs = data['probability']
    types = np.char.decode(data['type'])

    def load_cell_image(fname):
        with Image.open(fname) as image:
            return np.asarray(image)

    dir = os.path.dirname(fname)

    images = np.array([load_cell_image(os.path.join(dir, fn))
                       for fn in image_fnames])

    return image_fnames, probs, types

images, probs, types = load_dataset()
mono_images, mono_probs = [], []
poly_images, poly_probs = [], []
for image, prob, type in zip(images,probs, types):
    
    if type == "poly":
        poly_images.append(image)
        poly_probs.append(prob)
    else:
        mono_images.append(image)
        mono_probs.append(prob)


from sklearn.model_selection import train_test_split
from shutil import copy



def test_train_valid_split(X, y, path):
    
    #test train valid split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATIO, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=SPLIT_RATIO, random_state=1)
    
    # makeing the target folder
    if not os.path.exists(os.path.join(path)):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "train"))
        os.makedirs(os.path.join(path, "test"))
        os.makedirs(os.path.join(path, "val"))

    #
    with open(path + "/class.txt",'w') as f:
        upper = 2 if DROP36 else 4
        for i in range(0,upper):
            f.writelines(str(i)+'\n')

    os.makedirs(os.path.join(path + "/train", "0"))
    os.makedirs(os.path.join(path + "/train", "1"))
    os.makedirs(os.path.join(path + "/test", "0"))
    os.makedirs(os.path.join(path + "/test", "1"))
    os.makedirs(os.path.join(path + "/val", "0"))
    os.makedirs(os.path.join(path + "/val", "1"))
    
    if not DROP36:
        os.makedirs(os.path.join(path + "/val", "2"))
        os.makedirs(os.path.join(path + "/val", "3"))
        os.makedirs(os.path.join(path + "/train", "2"))
        os.makedirs(os.path.join(path + "/train", "3"))
        os.makedirs(os.path.join(path + "/test", "2"))
        os.makedirs(os.path.join(path + "/test", "3"))       
    with open(path + "/train.txt", 'w') as f:


        for x_insntance, y_instance in zip(X_train, y_train):
            ori_img_path = x_insntance
            if mapper_dict[y_instance] != 9:
                copy(ori_img_path, path + "/train/"+str(mapper_dict[y_instance]))
                f.write(str(mapper_dict[y_instance]) + "/" + x_insntance.split('/')[1] + " " + str(mapper_dict[y_instance]) + '\n')
    
    with open(path + "/test.txt", 'w') as f:

        for x_insntance, y_instance in zip(X_test, y_test):
            if mapper_dict[y_instance] != 9:
                ori_img_path = x_insntance
                copy(ori_img_path, path + "/test/"+str(mapper_dict[y_instance]))
                f.write(str(mapper_dict[y_instance]) + "/" + x_insntance.split('/')[1] + " " + str(mapper_dict[y_instance]) + '\n')

    with open(path + "/val.txt", 'w') as f:

        for x_insntance, y_instance in zip(X_val, y_val):
            if mapper_dict[y_instance] != 9:
                ori_img_path = x_insntance
                copy(ori_img_path,  path + "/val/"+str(mapper_dict[y_instance]))
                f.write(str(mapper_dict[y_instance]) + "/" + x_insntance.split('/')[1] + " " + str(mapper_dict[y_instance]) + '\n')

    return X_train, X_test, X_val, y_train, y_test, y_val

test_train_valid_split(poly_images, poly_probs, "poly")
test_train_valid_split(mono_images, mono_probs, "mono")