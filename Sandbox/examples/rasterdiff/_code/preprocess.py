import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import torch

# data array
images = []
for i in range(1, 15):
    img = cv2.imread(f"examples/rasterdiff/_images/{i}.exr", cv2.IMREAD_UNCHANGED)
    img = img[:,:,::-1]
    # extend to 4 channels
    img = np.concatenate([img, np.ones_like(img[...,0:1])], axis=-1)
    # to int8 type
    img = (img*255).astype('uint8')
    images.append(img)
images = np.array(images)

rgb32 = images.view(np.int32)
# as torch tensor
rgb = torch.from_numpy(rgb32).squeeze()
# save
torch.save(rgb, 'examples/rasterdiff/_data/gt.pt')

# billboard = np.array(billboard_list)
# rgb = torch.from_numpy(rgb32).squeeze()
# pose = torch.from_numpy(pose).squeeze()
# print(rgb.shape)
# print(pose.shape)
# # save
# torch.save(rgb, 'examples/nerf/_model/images.pt')

# print(images.shape)
# np.save("examples/rasterdiff/_data/gt.npy", images)