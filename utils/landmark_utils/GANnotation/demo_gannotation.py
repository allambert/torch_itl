import numpy as np
import GANnotation
import matplotlib.pyplot as plt
import utils
import cv2
import torch
import sys

"""
Original Code - Commented
myGAN = GANnotation.GANnotation(path_to_model='myGEN.pth', enable_cuda=False)
points = np.loadtxt('test_images/test_1.txt').transpose().reshape(66,2,-1)

image = cv2.cvtColor(cv2.imread('test_images/test_1.jpg'),cv2.COLOR_BGR2RGB)
image = image/255.0
image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
image = image.type_as(torch.FloatTensor())

images, cropped_pts = myGAN.reenactment(image,points[:,:,0:2])


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_1.avi',fourcc, 30.0, (128,128))
for imout in images:
    out.write(cv2.cvtColor(imout, cv2.COLOR_RGB2BGR))

out.release()
np.savetxt('test_1_cropped.txt', cropped_pts.reshape((132,-1)).transpose())
"""

# Set paths
# run demo_gannotation neutral_path, emotional_landmark_path, output_path
neu_im_path = sys.argv[1]
emo_lnd_path = sys.argv[2]
out_im_path = sys.argv[3]

# Set model
myGAN = GANnotation.GANnotation(path_to_model='myGEN.pth', enable_cuda=False)
# Read points and image
points = np.loadtxt(emo_lnd_path).reshape(68,2,-1)
points_concat = np.concatenate((points[0:60], points[61:64], points[65:]))

image = cv2.cvtColor(cv2.imread(neu_im_path),cv2.COLOR_BGR2RGB)
image = image/255.0
image = torch.from_numpy(image.swapaxes(2,1).swapaxes(1,0))
image = image.type_as(torch.FloatTensor())

images, cropped_pts = myGAN.reenactment(image,points_concat)
# Write output
cv2.imwrite(out_im_path, cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))

