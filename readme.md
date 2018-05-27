## Description of Files:

Each folder contains the following files:
- train : python script to train the model.
- test : python script to test the trained model.
- models.py : python class that defines Generator and discriminator models

## Folders

#### The folder \*_enhanced contains code for training using the Small model plus enhanced images as ground truth.

- Generator : Pretrained RESNET-18 ( 5 Basic Blocks)
- Discriminator : Pretrained VGG-13 + w/bn + Dense Layer + Leaky relu  + Dense Layer
- Pretrained layers were Frozen.
- Input image size 128*128
- Output image size 256*256
- Trained for 39 Epochs / around 2.5 days
- Trained on Ashwin's Laptop
- Dataset - Imagenet 8k
- Xavier Initialization was used.

We sharpened the "High_Res" real image and then made the GAN learn the mapping between "Low Res" Image and "High_Res_Sharpened" image. The results were that the GAN performed better than the Deep GAN as edges were more well-defined.

#### The folder \*_Small contains the architecture having only 5 basic blocks in the Generator.

- Generator : Pretrained RESNET-18 ( 5 Basic Blocks)
- Discriminator : Pretrained VGG-13 + w/bn + Dense Layer + Leaky relu  + Dense Layer
- Pretrained layers were Frozen.
- Input image size 128*128
- Output image size 256*256
- Trained for 50 Epochs / around 2.5 days
- Dataset - Imagenet 8k
- Xavier Initialization was used.

#### The folder \*_large contains the architecture for the bigger architecture with 13 basic blocks.
Deep GAN with
- Generator : Pretrained RESNET-34 ( 13 Basic Blocks)
- Discriminator : Pretrained VGG-13 + w/bn + Dense Layer + Leaku  relu  + Dense Layer
- None of the Pretrained layers were Frozen.
- Input image size 128*128
- Output image size 256*256
- Trained for 33 Epochs / around 3 days
- Trained on Google Cloud
- Dataset - Imagenet 8k
- Xavier Initialization was used.

## References
Code for this project was taken and modified from:
https://github.com/aitorzip/PyTorch-SRGAN
Architecture for the project was taken from:
C. Ledig and et. al. Photo-realistic single image super-resolution using a generative adversarial network. 2017.


