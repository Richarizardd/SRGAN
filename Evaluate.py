import math
import os

import numpy as np
import skimage
from skimage import measure



def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

SSIMs_pred_res, PSNRs_pred_res = [], []
SSIMs_resize_res, PSNRs_resize_res = [], []



path = "results/4_Residual_Test/"
test_dir = os.listdir(path)

for i in np.arange(0, len(test_dir), 3):
    if i % 60 == 0:
        print i, i+1, i+2
        
    name1, name2, name3 = test_dir[i].split("_")[0], test_dir[i+1].split("_")[0], test_dir[i+2].split("_")[0]
    assert name1 == name2 == name3
    
    gt = np.array(Image.open(path+test_dir[i]))
    resize = np.array(Image.open(path+test_dir[i+1]))
    pred = np.array(Image.open(path+test_dir[i+2]))
    
    SSIMs_pred_res.append(measure.compare_ssim(gt, pred, multichannel=True))
    SSIMs_resize_res.append(measure.compare_ssim(gt, resize, multichannel=True))
    
    PSNRs_pred_res.append(psnr(gt, pred))
    PSNRs_resize_res.append(psnr(gt, resize))

print "SSIM Pred:", np.mean(SSIMs_pred_res)
print "SSIM Resize:", np.mean(SSIMs_resize_res)

print "PSNR Pred:", np.mean(PSNRs_pred_res)
print "PSNR Resize:", np.mean(PSNRs_resize_res)



SSIMs_pred_dense, PSNRs_pred_dense = [], []
SSIMs_resize_dense, PSNRs_resize_dense = [], []

path = "4_Dense_Test/"
test_dir = os.listdir(path)

for i in np.arange(0, len(test_dir), 3)[:0]:
    if i % 60 == 0:
        print i, i+1, i+2
        
    gt = np.array(Image.open(path+test_dir[i]))
    resize = np.array(Image.open(path+test_dir[i+1]))
    pred = np.array(Image.open(path+test_dir[i+2]))
    
    SSIMs_pred_dense.append(measure.compare_ssim(gt, pred, multichannel=True))
    SSIMs_resize_dense.append(measure.compare_ssim(gt, resize, multichannel=True))
    
    PSNRs_pred_dense.append(psnr(gt, pred))
    PSNRs_resize_dense.append(psnr(gt, resize))

print "SSIM Pred:", np.mean(SSIMs_pred_dense)
print "SSIM Resize:", np.mean(SSIMs_resize_dense)

print "PSNR Pred:", np.mean(PSNRs_pred_dense)
print "PSNR Resize:", np.mean(PSNRs_resize_dense)