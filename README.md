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
- Vgg (vgg1, vgg13, vgg16)

### ResNet's Architectures and configurations:

```
name:  conv1
name:  bn1
name:  relu
name:  maxpool
name:  layer1
name:  layer2
name:  layer3
name:  layer4
name:  avgpool
name:  fc1
```

Cut-blocks parameters, e.g., 1 means that the network will be cut in the output of the layer1, and then connected to the avgpool.:
- [1, 2, 3, 4]

Fine-tuning layers parameters, from top to bottom, e.g., 1 means only the fc1 layer will be finetuned, and 10 that all the layers will be finetuned:
- If cut-block = 4, finetuning layers could be: [1, 3, 4, 5, 6, 10]
- If cut-block = 3, finetuning layers could be: [1, 3, 4, 5, 6, 9]
- If cut-block = 2, finetuning layers could be: [1, 3, 4, 5, 6, 8]
- If cut-block = 1, finetuning layers could be: [1, 3, 4, 5, 6, 7]

### Vgg's Architectures and configurations:

```
name:  conv1
name:  conv2
name:  conv3
name:  conv4
name:  conv5
name:  avgpool
name:  fc
```

Cut-blocks parameters, e.g., 1 means that the network will be cut in the output of the conv2, and then connected to the avgpool.:
- [1, 2, 3, 4]

Fine-tuning layers parameters, from top to bottom, e.g., 1 means only the fc1 layer will be finetuned, and 7 that all the layers will be finetuned:
- If cut-block = 4, finetuning layers could be: [1, 3, 4, 5, 6, 7]
- If cut-block = 3, finetuning layers could be: [1, 3, 4, 5, 6]
- If cut-block = 2, finetuning layers could be: [1, 3, 4, 5]
- If cut-block = 1, finetuning layers could be: [1, 3, 4]
- 
**For all models, if a -1 is passed as finetuning layer, all the layers will be finetuned.**



All the configuration parameters are in a text file, such as

```
[PARAMS]
DATA_DIR = stenosis_data
WEIGHTS_DIR = weights
model_name = resnet18
cut_block = 3
train_layers = -1
batch_size = 4
pretrained = True
lr = 0.001
momentum = 0.8
factor= 0.1
patience = 20
num_epochs = 100
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
- **state_dict_model_resnet18_03_-1.pth** : resnet18 weights with three residuals blocks (of four) with all the layers finetuned.

All the configuration parameters are in a text file, such as

```
[PARAMS]
DATA_DIR = stenosis_data
WEIGHTS_DIR = weights
model_name = resnet18
cut_block = 3
train_layers = -1
```


Running example (in a coolab jupyter notebook):

```python
%run main_test.py test_params.txt
```

It generates in the *WEIGHTS_DIR* folder (where the weights are loaded) 
- A *.json* file with the testing report from each metric
- A *.csv* file with the testing filename of the image, the ground-truth label, the predicted probability, and the predicted label, respectively. 

**Best ResNet Detection results** (may vary from the previously reported results):

| Model        | Pretrained | Cut-block | Fine-tuning layer  | ACC | Prec | Rec  | F1    | Spec | 
| ------------ |----------|---------|------------------|---|----|---|---|---| 
|    ResNet18          |       True     |    3       |         -1           |   **0.9426**  | **0.9642**     |  **0.9152**    |    **0.9391**   |  **0.9682**   |
|    ResNet18      |       True     |     2    |         1           |    0.9098  |  0.9444 |  0.8644  |   0.9026   |  0.9523  |
|    ResNet18          |       False     |      2    |         -1           |  0.9098 |  0.9285   |   0.8813  |   0.9043    |  0.9365  |
|    ResNet34          |       True     |    3       |         -1           |   0.9344  | 0.9636     | 0.8983    |    0.9298   |  0.9682   |
|    ResNet34      |       True     |      2   |         1           |   0.9344  |  0.9473 |   0.9152 |   0.9310    |  0.9523  |
|    ResNet34          |       False     |     1     |         -1           |  0.9098  |  0.9444   |   0.8644  |   0.9026   |  0.9523  |
|    ResNet50      |       True     |     3      |         -1           |  0.9344   |   0.9473 | 0.9152   |  0.9310    |  0.9523  |
|    ResNet50      |       True     |     2     |         1           |  0.9016   |  0.9607  |  0.8305  |  0.8909    | 0.9682   |
|    ResNet50         |       False     |       2   |         -1           |  0.9098 |   0.9285  |  0.8813   |    0.9043   |  0.9365  |

**>If -1:all layers were finetuned, 1:only the fc1 was finetuned (CNN as feature extractor only)**

**Best Vgg Detection results** (may vary from the previously reported results):

**TBA**


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

