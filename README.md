# SRGAN for Endoscopy 

## Train

To train ResNet, run:
```
python train.py
```


## Testing

To test ResNet + DenseNet respectively on the Testing dataset, run:
```
python test.py --dataroot ./datasets/Test/  --gpu_ids 0 --name 4_Residual_Test --which_model_netG Residual
python test.py --dataroot ./datasets/Test/  --gpu_ids 0 --name 4_Dense_Test --which_model_netG Dense
```
--dataroot = path to dataset.
--gpu_ids = which GPU to test with (-1 means CPU mode)
--name = title of the experiment you want to run (creates a folder in ./results)
--which_model_netG = which model to run (atm, only ResNet + DenseNet are implemented)

To test on an individual image (162a.jpg), run:
```
python2 test_image.py --image_name "datasets/Test/162a.jpg" --model_name "Dense_netG_epoch_4_100.pth"
```

## Evaluating
```
python Evaluate.py
```

Slight errors atm, bc the testing images are a pixel bigger than the original, causing PSNR to not eval correctly
