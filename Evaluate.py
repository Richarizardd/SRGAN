import math, os, re, sys

import numpy as np
from PIL import Image
import skimage
from skimage import measure

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def evaluate(path):
    PSNRs_resize, PSNRs_pred = [], []
    SSIMs_resize, SSIMs_pred = [], []

    test_dir = os.listdir(path)
    test_dir.sort()

    for i in np.arange(0, len(test_dir), 3):
        # if i % 60 == 0: print i, i+1, i+2

        gt = np.array(Image.open(path+test_dir[i]))
        pred = np.array(Image.open(path+test_dir[i+1]))
        resize = np.array(Image.open(path+test_dir[i+2]))

        SSIMs_pred.append(measure.compare_ssim(gt, pred, multichannel=True))
        SSIMs_resize.append(measure.compare_ssim(gt, resize, multichannel=True))

        PSNRs_pred.append(psnr(gt, pred))
        PSNRs_resize.append(psnr(gt, resize))

    print "SSIM Pred:", np.mean(SSIMs_pred)
    print "SSIM Resize:", np.mean(SSIMs_resize)

    print "PSNR Pred:", np.mean(PSNRs_pred)
    print "PSNR Resize:", np.mean(PSNRs_resize)
    
def main():
    path = sys.argv[1]
    evaluate(path)

if __name__ == "__main__":
    main()