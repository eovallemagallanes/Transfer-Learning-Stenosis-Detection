# utils functions for networks model configurations
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.ResNet import ResidualNet
from models.VGGNet import VggNet


# set to trainable a set of layers
def set_parameter_requires_grad(model_ft, num_trainable_layers=0):
    layers = list(model_ft._modules.keys())

    if num_trainable_layers > len(layers) or num_trainable_layers < 0:
        raise ValueError(
            "num_trainable_layers should be an integer between 0 (all freeze) to : {} (all trainables):  ".format(
                len(layers)))
    # FREEZING LAYERS
    total_children = 0
    children_counter = 0
    num_trainable_layer = 0
    total_freeze = 0
    total_trainable = 0
    for c in model_ft.children():
        total_children += 1

    if num_trainable_layers == 0:
        num_trainable_layer = total_children

    for c in model_ft.children():
        if children_counter < total_children - num_trainable_layers:
            for param in c.parameters():
                param.requires_grad = False
            total_freeze += 1
        else:
            for param in c.parameters():
                param.requires_grad = True
            total_trainable += 1
        children_counter += 1

    print('Total trainable: {}, Total freeze: {}'.format(total_trainable, total_freeze))

    return model_ft


def transform(num_output_channels=3, normalize=True):
    tGrayscale = transforms.Grayscale(num_output_channels=num_output_channels)

    if normalize:
        data_transforms = transforms.Compose(
            [
                tGrayscale,
                transforms.ToTensor(),
                # Normalize input channels using mean values and standard deviations of ImageNet.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                tGrayscale,
                transforms.ToTensor()
            ]
        )

    return data_transforms


def createTrainValDataLoaders(DATA_DIR, batch_size, num_output_channels=3, normalize=True):
    # create dataloaders
    data_transforms = transform(num_output_channels=num_output_channels, normalize=normalize)

    train_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), data_transforms)
    train_loader = DataLoader(dataset=train_image_datasets, shuffle=True, batch_size=batch_size)

    val_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "validation"), data_transforms)
    val_loader = DataLoader(dataset=val_image_datasets, shuffle=False, batch_size=batch_size)

    # merge train & val data loaders
    dataloaders = {'train': train_loader, 'validation': val_loader}
    dataset_sizes = {'train': len(train_image_datasets), 'validation': len(val_image_datasets)}

    return dataloaders, dataset_sizes


def createTestDataLoaders(DATA_DIR, batch_size=1, num_output_channels=3, normalize=True):
    # create dataloaders
    data_transforms = transform(num_output_channels=num_output_channels, normalize=normalize)
    # create dataloaders
    test_image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), data_transforms)
    test_loader = DataLoader(dataset=test_image_datasets, shuffle=False, batch_size=batch_size)

    return test_loader


def createModel(model_type, model_deep, num_blocks, num_trainable_layers=-1, pretrained=True, num_classes=1):
    model_name = '%s%d' % (model_type, model_deep)
    if model_type == 'vgg':
        model = VggNet(model_name=model_name, num_blocks=num_blocks, num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'resnet':
        model = ResidualNet(model_name=model_name, num_blocks=num_blocks, num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError("model not found, available models: vgg, resnet")

    # freezing layers
    if num_trainable_layers == -1:
        print('All layers set to trainable')
    else:
        print('A total of {} layers are set to train'.format(num_trainable_layers))
        model = set_parameter_requires_grad(model, num_trainable_layers)

    return model
