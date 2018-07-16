# SRGAN for Endoscopy 

## Testing

To test ResNet + DenseNet respectively, run:
```
python test.py --dataroot ./datasets/Test/  --gpu_ids 0 --name 4_Residual_Test --which_model_netG Residual
python test.py --dataroot ./datasets/Test/  --gpu_ids 0 --name 4_Dense_Test --which_model_netG Dense
```
--dataroot = path to dataset.
--gpu_ids = which GPU to test with (-1 means CPU mode)
--name = title of the experiment you want to run (creates a folder in ./results)
--which_model_netG = which model to run (atm, only ResNet + DenseNet are implemented)

## Evaluating
```
python Evaluate.py
```

Slight errors atm, bc the testing images are a pixel bigger than the original, causing PSNR to not eval correctly
