import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def imshow(inp, figsize, filename=None, imagenet=True):
    """Imshow for Tensor."""
    plt.figure(figsize=figsize)

    inp = inp.numpy().transpose((1, 2, 0))
    if imagenet:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.001)  # pause a bit so that plots are updated

    if filename:
        plt.imsave(filename,inp, format='pdf')


def imsave(imgs, filepath, filenames, imagenet=True):
    for img, filename in zip(imgs, filenames):
        filename = filename[0][-7:-4] # just the image index (3 digits) e.g 001, 180
        img = img.numpy().transpose((1, 2, 0))
        if imagenet:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
        img = np.clip(img, 0, 1)
        img = np.uint8(img*255) # need to be in uint8 from 0 to 255
        pil_img = Image.fromarray(img)
        pil_img.save('%s/%s.png'%(filepath, filename))