import torch
from torchvision import datasets, transforms

import os
import sys
import configparser


from models.ResNet import ResidualNet
from test import test_model, eval_preds

# set manual seed
torch.manual_seed(42)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read parameters
args = str(sys.argv)
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# read configuration parameters
config = configparser.ConfigParser()
config.read(sys.argv[1])

DATA_DIR = config.get('PARAMS', 'DATA_DIR')
WEIGHTS_DIR = config.get('PARAMS', 'WEIGHTS_DIR')
model_name = config.get('PARAMS', 'model_name')
cut_block = int(config.get('PARAMS', 'cut_block'))
train_layers = int(config.get('PARAMS', 'train_layers'))

print('*'*50)
print('Model: {}'.format(model_name))
print('Cut block: {}'.format(cut_block))
print('Num train_layers: {}'.format(train_layers))

# define data transformations
data_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # Normalize input channels using mean values and standard deviations of ImageNet.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# create dataloaders
test_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), data_transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_image_datasets, shuffle=False, batch_size=1)


# create model
model = ResidualNet(model_name= model_name, num_blocks=cut_block, num_classes=1, num_trainable_layers=-1, pretrained=False)
model.to(device)

# load weights
model_weights = '%s/state_dict_model_%s_%02d_%02d.pth' %(WEIGHTS_DIR, model_name, cut_block, train_layers)
pretrained_checkpoint = torch.load(model_weights)
model.load_state_dict(pretrained_checkpoint['model_state_dict'])
model.eval()

# test
results_report = '%s/model_results_%s_%02d_%02d.json' %(WEIGHTS_DIR, model_name, cut_block, train_layers)
results_report_probas = '%s/result_probas_model_%s_%02d_%02d.csv' %(WEIGHTS_DIR, model_name, cut_block, train_layers)

print('*'*50)
print('Testing started:')
y_true, y_pred = test_model(device=device, model=model, test_loader=test_loader, PATH_RESULTS=results_report_probas)
eval_preds(y_true, y_pred, results_report)




