# Transfer-Learning-Stenosis-Detection
----------

In this repository you can find the code implementation for the paper "Transfer Learning for Stenosis Detection in X-ray Coronary Angiography", 
published on Mathematics 2020, Volume 8, Issue 9, 1510: https://www.mdpi.com/817954

This paper presents a new method for the automatic detection of coronary artery stenosis in XCA images, employing a pre-trained Convolutional Neural Network (CNN) via Transfer Learning. The method is based on a network-cut and fine-tuning approach.

----------
## Release notes

All the experiments were computed on a Cloud-Platform with an Intel (R) Xeon (R), 12 GB of RAM, and a 2.00 GHz dual processor, and a Tesla P4 GPU with 8 GB VRAM. The algorithms were implemented in Pytorch 1.8 library and Python 3.7.10.

The source code is in constant update.  Stay close to updates.
- 2021.04: Release release/1.8-pytorch version

Aditional packages: torchsummary and livelossplot.

----------
## Training stage

The training XCA images were patches of <img src="https://render.githubusercontent.com/render/math?math=32 \times 32"> pixels, such that

- stenosis_data
    - train
        - positive
        - negative
    - validation
        - positive
        - negative

Supported architectures (so far):
- ResNet (resnet18, resnet34, resnet50)
- VGG (vgg11, vgg13, vgg16)

### ResNet's Architectures and configurations:

```
(0a):  conv1
(0b):  bn1
(0c):  relu
(0d):  maxpool
(1):  layer1
(2):  layer2
(3):  layer3
(4):  layer4
(pooling):  avgpool
(fc):  fc1
```

Cut-blocks parameters, e.g., 1 means that the network will be cut in the output of the layer1 (fisrt residual block), and then connected to the avgpool.:
- [1, 2, 3, 4]

Fine-tuning layers parameters, from top to bottom, e.g., 1 means only the fc1 layer will be finetuned.


### Vgg's Architectures and configurations:

```
(0):  conv1
(1):  conv2
(2):  conv3
(3):  conv4
(4):  conv5
(pooling):  avgpool
(fc):  fc
```

Cut-blocks parameters, e.g., 1 means that the network will be cut in the output of the conv2 (first convolutional block), and then connected to the avgpool.:
- [1, 2, 3, 4]

Fine-tuning layers parameters, from top to bottom, e.g., 1 means only the fc1 layer will be finetuned, and 7 that all the layers will be finetuned:

**For all models, if a -1 is passed as finetuning layer, all the layers will be finetuned.**



All the configuration parameters are in a text file, such as

```
[PARAMS]
DATA_DIR = 'DATASET FOLDER'
WEIGHTS_DIR = 'DIRECTORY TO SAVE WEIGHTS '
torch_seed = 'PYTORCH SEED'
model_type = 'MODEL TYPE: resnet/vgg'
model_deep = 'MODEL DEPTH'
model_name = 'MODEL FILENAME'
cut_block = 'CUT BLOCK: 1-4'
train_layers = 'NO. TRAINABLE LAYERS'
batch_size = 'TRAINING BATCH SIZE'
pretrained = 'USE IMAGENET: True/False'
lr = 'LEARNING RATE'
momentum = 'MOMENTUM
factor= 'LEARNING RATE DECAY FACTOR'
patience = 'VAL_LOSS PATIENCE'
num_epochs = 'NUM OF EPOCHS TO TRAIN'
finetuning = 'FINETUNE A PREVIOS MODEL, READED FROM WEIGHTS_DIR'
```


Running example (in a coolab jupyter notebook):

```python
%run main_train.py trian_params.txt
```

It generates in the *WEIGHTS_DIR* folder:
- A *.pth* file with the weights of the model
- A *.json* file with the training acc/loss history

----------
## Testing stage

If you want to test in your own dataset, set on a train folder all the XCA images. Notice that the size **must be** <img src="https://render.githubusercontent.com/render/math?math=32 \times 32"> pixels, and separate into two subfolders: positive and negatives cases. For instance:
- stenosis_data
    - train
        - positive
        - negative

Weights example: 
- **model_resnet18_3RB_ALL_FT.pth** : resnet18 weights with three residuals blocks (of four) with all the layers finetuned.

All the configuration parameters are in a text file, such as

```
[PARAMS]
DATA_DIR = 'DATASET PATH'
WEIGHTS_DIR = 'DIRECTORY TO SAVE WEIGHTS '
model_type = 'MODEL TYPE: resnet/vgg'
model_deep = 'MODEL DEPTH'
model_name = 'MODEL FILENAME'
cut_block = 'CUT BLOCK: 1-4'
train_layers = -1
RESULTS_DIR = 'RESULTS PATH'
GRADCAM = 'APPLY GRADCAM: True/False'
imagenet_norm = 'APPLY IMAGENET NORMALIZATION: True/False'
cam_layer = 'GRADCAM LAYER'
```


Running example (in a coolab jupyter notebook):

```python
%run main_test.py test_params.txt
```

It generates in the *WEIGHTS_DIR* folder (where the weights are loaded) 
- A *.json* file with the testing report from each metric
- A *.csv* file with the testing filename of the image, the ground-truth label, the predicted probability, and the predicted label, respectively. 

**Detection results** may change from the previously reported results due to the random seed, random split test/validation/train dataset:


----------
## Cite as

If you use this for research, please cite. Here is an example BibTeX entry:

```
@article{ovalle2020transfer,
  title={Transfer Learning for Stenosis Detection in X-ray Coronary Angiography},
  author={Ovalle-Magallanes, Emmanuel and Avina-Cervantes, Juan Gabriel and Cruz-Aceves, Ivan and Ruiz-Pinales, Jose},
  journal={Mathematics},
  volume={8},
  number={9},
  pages={1510},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
----------
## Development

Want to contribute? Great!. Contact us.

----------
## License

MIT

