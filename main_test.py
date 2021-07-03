import torch

import sys
import configparser

from test import test_model, eval_preds
from models.utilsNet import createModel, createTestDataLoaders
from gradcam.GradCam import GradCam, computeGradCam


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read parameters
args = str(sys.argv)
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# read configuration parameters
config = configparser.ConfigParser()
config.read(sys.argv[1])

# weights dir
DATA_DIR = config.get('PARAMS', 'DATA_DIR')
WEIGHTS_DIR = config.get('PARAMS', 'WEIGHTS_DIR')

# model configuration
model_type = config.get('PARAMS', 'model_type')
model_deep = int(config.get('PARAMS', 'model_deep'))
model_name = config.get('PARAMS', 'model_name')
cut_block = int(config.get('PARAMS', 'cut_block'))
train_layers = int(config.get('PARAMS', 'train_layers'))
imagenet_norm = True if config.get('PARAMS', 'imagenet_norm') == 'True' else False
cam_layer = config.get('PARAMS', 'cam_layer')

# results dir
RESULTS_DIR = config.get('PARAMS', 'RESULTS_DIR')
GRADCAM = True if config.get('PARAMS', 'GRADCAM') == 'True' else False

print('*'*50)
print('Model: {}'.format(model_name))
print('Cut block: {}'.format(cut_block))
print('Num train_layers: {}'.format(train_layers))

# create dataloaders
# if imagenet_norm, apply ImageNet normalization
test_loader = createTestDataLoaders(DATA_DIR, batch_size=1, num_output_channels=3, normalize=imagenet_norm)

# create model
model = createModel(model_type=model_type, model_deep=model_deep, num_blocks=cut_block, num_trainable_layers=train_layers,
                    pretrained=False, num_classes=1)
model.to(device)

# load weights
model_weights = '%s/model_%s.pth' %(WEIGHTS_DIR, model_name)
pretrained_checkpoint = torch.load(model_weights,  map_location=device)
model.load_state_dict(pretrained_checkpoint['model_state_dict'])
#model.eval()

# test
results_report = '%s/model_results_%s.json' %(RESULTS_DIR, model_name)
results_report_probas = '%s/result_probas_model_%s.csv' %(RESULTS_DIR, model_name)

print('*'*50)
print('Testing started:')
y_true, y_pred = test_model(device=device, model=model, test_loader=test_loader, PATH_RESULTS=results_report_probas)
eval_preds(y_true, y_pred, results_report)

# show model parent layers
#for name, _ in model.named_children():
#    print(name)

if GRADCAM:
    gradcam = GradCam(model=model, cam_layer_name=cam_layer, imagenet_norm=imagenet_norm)
    computeGradCam(gradcam, test_loader, device, RESULTS_DIR, model_name)






