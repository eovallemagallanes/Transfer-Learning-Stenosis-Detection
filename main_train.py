import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import sys
import configparser

from train import train_model
from models.utilsNet import createTrainValDataLoaders, createModel

from torchsummary import summary

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read parameters
args = str(sys.argv)
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# read configuration parameters
config = configparser.ConfigParser()
config.read(sys.argv[1])

# data IO path
DATA_DIR = config.get('PARAMS', 'DATA_DIR')
WEIGHTS_DIR = config.get('PARAMS', 'WEIGHTS_DIR')

# model architecture configuration
torch_seed = int(config.get('PARAMS', 'torch_seed'))
model_type = config.get('PARAMS', 'model_type')
model_deep = int(config.get('PARAMS', 'model_deep'))
model_name = config.get('PARAMS', 'model_name')
cut_block = int(config.get('PARAMS', 'cut_block'))
train_layers = int(config.get('PARAMS', 'train_layers'))

# load imagenet weights
pretrained = True if config.get('PARAMS', 'pretrained') == 'True' else False

# hyperparameters
lr = float(config.get('PARAMS', 'lr'))
momentum = float(config.get('PARAMS', 'momentum'))
factor = float(config.get('PARAMS', 'factor'))
patience = int(config.get('PARAMS', 'patience'))
batch_size = int(config.get('PARAMS', 'batch_size'))
num_epochs = int(config.get('PARAMS', 'num_epochs'))

# load a pre-trained model flag
finetuning = True if config.get('PARAMS', 'finetuning') == 'True' else False

# set manual seed
torch.manual_seed(torch_seed)

print('*' * 50)
print('Model type: ', model_type)
print('Deep: ', model_deep)
print('Cut block: ', cut_block)
print('Num train_layers: ', train_layers)
print('Pretrained: ', pretrained)
print('Finetuning: ', finetuning)

# create dataloaders
# if pretrained, apply ImageNet normalization
dataloaders, dataset_sizes = createTrainValDataLoaders(DATA_DIR, batch_size, 3, pretrained)


# create model
model = createModel(model_type=model_type, model_deep=model_deep, num_blocks=cut_block, num_trainable_layers=train_layers,
                    pretrained=pretrained, num_classes=1)
model.to(device)

summary(model, (3, 32, 32))


if finetuning:
    model_weights = '%s/state_dict_model_%s.pth' % (WEIGHTS_DIR, model_name)
    model_name = model_name + '_finetuning'
    print('Loading weights from ', model_weights)
    pretrained_checkpoint = torch.load(model_weights,  map_location=device)
    model.load_state_dict(pretrained_checkpoint['model_state_dict'])

# weights path
model_checkpoint = '%s/state_dict_model_%s.pth' % (WEIGHTS_DIR, model_name)
model_history = '%s/model_history_%s.json' % (WEIGHTS_DIR, model_name)

# set optimizer and lr-scheduler
criterion = nn.BCEWithLogitsLoss(reduction='sum').cuda()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=factor, patience=patience,
                                           verbose=True)

# train the model
print('*' * 50)
model, m_history = train_model(device=device, model=model, criterion=criterion,
                               optimizer=optimizer, scheduler=scheduler,
                               num_epochs=num_epochs, batch_size=batch_size,
                               dataloaders=dataloaders,
                               dataset_sizes=dataset_sizes,
                               PATH_MODEL=model_checkpoint,
                               PATH_HISTORY=model_history,
                               show_plot=False)
