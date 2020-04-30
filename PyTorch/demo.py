#######################################################################################################################
# Copyright(c) 2019 - present, Mahmoud Afifi
# York University, Canada
# Email: mafifi @ eecs.yorku.ca - m.3afifi @ gmail.com
#######################################################################################################################
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# Please cite the following work if this program is used:
# Mahmoud Afifi and Michael S.Brown.Sensor Independent Illumination Estimation for DNN Models. In BMVC, 2019
#######################################################################################################################


from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import os
import SIIE_torch as ssie


image_name = os.path.join('..', 'imgs_w_normalization', 'Cube+_challenge_CanonEOS550D_243.png') #image name
#Note: be sure that the image is in the raw-RGB linear space and
# the black/saturation normalization is correctly applied to the image before using it.

model_name = 'trained_model_wo_NUS_Canon1DsMkIII' #trained model name
in_img_sz = 150 # our network accepts 150x150 raw-RGB image

I = Image.open(image_name) # read the image

width, height = I.size # get size
if width != in_img_sz or height != in_img_sz:
    I_ = np.asarray(I.resize((in_img_sz, in_img_sz))) # resize the image
else:
    I_ = np.asarray(I)

I_ = I_.astype(np.float32)/255


#load model
SSIE = ssie.SIIE_torch()

SSIE = SSIE.loadModel(os.path.join('models', model_name))

# estimate illuminant
est_ill, mapped_image = SSIE.predict(model_name, I_)

est_ill = est_ill / np.linalg.norm(est_ill) # make it a unit vector
print('Estimated scene illuminant =  %f, %f, %f\n' % (est_ill[0], est_ill[1], est_ill[2])) # display the result

factor = 4 # scale factor to aid visualization

I = np.asarray(I)

input_image = Image.fromarray(np.uint8(I*factor))

mapped_image = Image.fromarray((mapped_image * 255 * factor).astype(np.uint8))
mapped_image = mapped_image.resize((width, height))
white_balanced_image = Image.fromarray(np.uint8(np.reshape(np.dot(np.reshape(I.astype(np.float32), [-1, 3]),
                            np.diag(est_ill[1]/est_ill)), [height, width, 3]) * \
                                factor))  # apply white balance correction then show the result (scaled to aid visualization)


fig = plt.figure()
figure = fig.add_subplot(1, 3, 1)
plt.imshow(input_image)
figure.set_title('Input image')
figure = fig.add_subplot(1, 3, 2)
plt.imshow(mapped_image)
figure.set_title('Mapped image')
figure = fig.add_subplot(1, 3, 3)
plt.imshow(white_balanced_image)
figure.set_title('White balanced')
plt.show()



