import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import matplotlib.pyplot as plt

# load exr images
path = "S:/SIByL2024/Sandbox/examples/nee/_image/scene_1"

def load_image(file_name):
    img = cv2.imread(f"{path}/{file_name}", cv2.IMREAD_UNCHANGED)[:,:,[2,1,0,3]]
    # set nan to 0
    img[img == np.nan] = 0
    # set inf to 0
    img[img == np.inf] = 0
    return img

# compute mse
def mse(a, b):
    return np.mean((a-b)**2)

gt = load_image("gt.exr")

lum = []
ois = []
slights = []
stratified = []
cv = []
wis = []

def measure_frames(frame_count):    
    lum1 = load_image(f"lum_{frame_count}frames.exr")
    ois1 = load_image(f"ois_{frame_count}frames.exr")
    slights1 = load_image(f"slights_{frame_count}frames.exr")
    stratified_1 = load_image(f"stratified_{frame_count}frames.exr")
    cv1 = load_image(f"rcv_{frame_count}frames.exr")
    wis_1 = load_image(f"wis_{frame_count}frames.exr")

    plt.imshow(wis_1)
    plt.show()

    print("MSE lum1:", mse(gt, lum1))
    print("MSE ois1:", mse(gt, ois1))
    print("MSE slights1:", mse(gt, slights1))
    print("MSE stratified_1:", mse(gt, stratified_1))
    print("MSE cv1:", mse(gt, cv1))
    print("MSE wis_1:", mse(gt, wis_1))

    lum.append(np.log(mse(gt, lum1)))
    ois.append(np.log(mse(gt, ois1)))
    slights.append(np.log(mse(gt, slights1)))
    stratified.append(np.log(mse(gt, stratified_1)))
    cv.append(np.log(mse(gt, cv1)))
    wis.append(np.log(mse(gt, wis_1)))

measure_frames(1)
measure_frames(5)
measure_frames(10)

plt.plot(lum, label="Lum")
plt.plot(ois, label="OIS")
plt.plot(slights, label="S-Lights")
plt.plot(stratified, label="Stratified")
plt.plot(cv, label="CV")
plt.plot(wis, label="WIS")
plt.legend()

plt.show()