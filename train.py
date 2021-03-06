import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
# from model import Generator, Discriminator
from Model_Residual import Generator, Discriminator
from Model_Dense import Generator as Dense


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--name', default='4_Residual_Test', type=str, help='generator model epoch name')
parser.add_argument('--which_model_netG', type=str, default="Residual")
#parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
#parser.add_argument('--which_epoch', default='100', type=str, help='which epoch')


parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--debug', default=1, type=int, help='debug')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

train_set = TrainDatasetFromFolder('datasets/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder('datasets/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True) #64
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
# for param in netG.parameters():
#     print("---------")
#     print(param.size())
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

# optimizerG = optim.Adam(netG.parameters(), weight_decay=1e-5)
# optimizerD = optim.Adam(netD.parameters())

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


def test():
    if os.path.isdir("./results/"+opt.name):
        os.system("rm -r ./results/"+opt.name)
    os.system('mkdir ./results/'+opt.name)

    for i, image_name in enumerate(os.listdir('datasets/val/')):
        #print i+1, "Test+Save:", "results/"+opt.name+"/"+image_name

        img_gt = Image.open('datasets/val/'+image_name).convert('RGB')
        
        if opt.which_model_netG == "Dense":
            max_width = 800
            if img_gt.size[0] > max_width:
                new_height = int(max_width/float(img_gt.size[0])*img_gt.size[1])
                new_height = int(math.floor(new_height / 2.) * 2)
                img_gt = img_gt.resize((max_width, new_height))

        img_test = img_gt.resize((img_gt.size[0]/opt.upscale_factor, img_gt.size[1]/opt.upscale_factor))
        img_test_tensor = Variable(ToTensor()(img_test), volatile=True).unsqueeze(0)
        img_test_tensor = img_test_tensor.cuda(opt.gpu_ids[0])
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

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z)
        # print("NetG Output: ", fake_img.size())
        # print("Real image: ", real_img.size())

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.data[0] * batch_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.data[0] * batch_size
        running_results['d_score'] += real_out.data[0] * batch_size
        running_results['g_score'] += fake_out.data[0] * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    netG.eval()
    out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []

    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = Variable(val_lr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        # print("LR: ", lr.size()) #torch.Size([1, 3, 72, 72])
        sr = netG(lr)

        # print("SR: ", sr.size())
        # print("HR: ", hr.size())
        # Idead Size torch.Size([1, 3, 288, 288])

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).data[0]
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        val_images.extend(
            [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 15)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    
    # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % opt.debug == 0 and epoch != 0:
        print "Testing Val...",
        test()
        #os.system("python test.py --dataroot ./datasets/val/  --gpu_ids 0 --name 4_Residual_Test --which_model_netG Residual")

    if epoch % 10 == 0 and epoch != 0:
        out_path = 'statistics/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                  'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

    # if epoch == NUM_EPOCHS:
    #     torch.save(netG, 'epochs/netG_model.pth')
