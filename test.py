import argparse
import math
import os
import time

import cv2
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from Model_Residual import Generator as Residual
from Model_Dense import Generator as Dense
from Evaluate import evaluate
# python test.py --dataroot ./datasets/Test/  --gpu_ids 0,1 --name 4_Residual_Test --which_model_netG Residual
# python test.py --dataroot ./datasets/Test/  --gpu_ids 0,1 --name 4_Dense_Test --which_model_netG Dense
# python test.py --dataroot ./datasets/Example/ --gpu_ids -1 --name 4_Dense_HD_Endoscopy --which_model_netG Dense
# python test.py --dataroot ./datasets/Example/ --gpu_ids -1 --name 4_Dense_HD_Endoscopy --which_model_netG Residual

### 1. Argparse + CUDA Initialization
parser = argparse.ArgumentParser(description='Test Dataset')
parser.add_argument('--dataroot', default='./datasets/val', type=str, help='test low resolution images')
parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--name', default='4_Residual_Test', type=str, help='generator model epoch name')
parser.add_argument('--which_model_netG', type=str, default="Residual")
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--which_epoch', default='100', type=str, help='which epoch')
opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
	id = int(str_id)
	if id >= 0:
		opt.gpu_ids.append(id)

if len(opt.gpu_ids) > 0:
	torch.cuda.set_device(opt.gpu_ids[0])

if os.path.isdir("./results/"+opt.name):
	os.system("rm -r ./results/"+opt.name)
os.system("mkdir ./results/"+opt.name)



### 2. Model Initialization
model = eval(opt.which_model_netG)(opt.upscale_factor).eval()

model_name = 'epochs/'+opt.which_model_netG+"_netG_epoch_"+str(opt.upscale_factor)+"_"+opt.which_epoch+".pth"
if len(opt.gpu_ids) > 0:
	print "Successfully loaded model in GPU Mode with:", opt.gpu_ids
	model.load_state_dict(torch.load(model_name))
	model.cuda(opt.gpu_ids[0])
else:
	print "Successfully loaded model in CPU Mode"
	model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

if len(opt.gpu_ids) > 0:
	model.to(opt.gpu_ids[0])
	model = torch.nn.DataParallel(model, opt.gpu_ids)

### 3. Test Procedure
if os.path.isdir("./results/"+opt.name):
	os.system("rm -r ./results/"+opt.name)
os.system('mkdir ./results/'+opt.name)

for i, image_name in enumerate(os.listdir(opt.dataroot+"/")):
	#print i+1, "Test+Save:", "results/"+opt.name+"/"+image_name

	img_gt = Image.open(opt.dataroot+"/"+image_name).convert('RGB')
	
	if opt.which_model_netG == "Dense":
		max_width = 800
		if img_gt.size[0] > max_width:
			new_height = int(max_width/float(img_gt.size[0])*img_gt.size[1])
			new_height = int(math.floor(new_height / 2.) * 2)
			img_gt = img_gt.resize((max_width, new_height))

	img_test = img_gt.resize((img_gt.size[0]/opt.upscale_factor, img_gt.size[1]/opt.upscale_factor))
	img_test_tensor = Variable(ToTensor()(img_test), volatile=True).unsqueeze(0)
	if len(opt.gpu_ids) > 0: img_test_tensor = img_test_tensor.cuda(opt.gpu_ids[0])
	out = model(img_test_tensor)
	img_pred = ToPILImage()(out[0].data.cpu())
	size = img_pred.size
	img_resize = img_test.resize((size[0], size[1]))
	img_gt = img_gt.resize((size[0], size[1]))
	assert img_gt.size == img_pred.size == img_resize.size
	img_pred.save("results/"+opt.name+"/"+image_name[:-4]+"_pred.jpg")
	img_resize.save("results/"+opt.name+"/"+image_name[:-4]+"_resize.jpg")
	img_gt.save("results/"+opt.name+"/"+image_name[:-4]+"_gt.jpg")

print "Evaluating...",
evalute('./results/'+opt.name)
print "Done"
