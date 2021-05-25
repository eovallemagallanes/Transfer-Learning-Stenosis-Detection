import time
import os
import copy
import json
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

# Plotting
import matplotlib.pyplot as plt
from livelossplot import PlotLosses

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    

def train_model(device, model, criterion, optimizer, scheduler, num_epochs, batch_size, dataloaders, dataset_sizes, PATH_MODEL, PATH_HISTORY, show_plot = False):
    liveloss = PlotLosses()

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    print("Training started:")

    hist_train = {'acc' : [],
                  'val_acc' : [],
                  'loss' : [],
                  'val_loss' : []
                  }

    for epoch in range(num_epochs):
        logs = {}
        #print('Epoch: {}/{}:'.format(epoch + 1, num_epochs), end=' ', flush=True,)

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = dataset_sizes[phase] // batch_size
            it = 0
            for inputs, labels in dataloaders[phase]:
                since_batch = time.time()
                batch_size_ = len(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                labels = labels.unsqueeze(1)
                labels = labels.float()
                
                optimizer.zero_grad()

                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    sigmoid = nn.Sigmoid()
                    preds = sigmoid(outputs) #sigmoid(outputs.cpu().detach().numpy())[:, 0]
                    preds = torch.tensor([1.0 if pred > 0.5 else 0.0 for pred in preds]).to(device)
                    preds = torch.reshape(preds, (batch_size_, 1))
                    
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                #print('preds shape: {}. true shape: {}'.format(preds.shape, labels.data.shape))
                #print('preds: {}. true: {}'.format(preds, labels.data))
                # Print iteration results
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #print('corrects in batch: {}'.format(torch.sum(preds == labels.data)))
                
                

                print(
                    "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                        phase,
                        epoch + 1,
                        num_epochs,
                        it + 1,
                        n_batches + 1,
                        time.time() - since_batch,
                    ),
                    end="\r",
                    flush=True,
                )
                
                it += 1

            # Print epoch results
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(
                    "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                        "train" if phase == "train" else "validation  ",
                        epoch + 1,
                        num_epochs,
                        epoch_loss,
                        epoch_acc,
                    )
                )
            

            # saving loss & accuracy
            if phase == 'train':
                hist_train['loss'].append(epoch_loss.cpu().float().item())
                hist_train['acc'].append(epoch_acc.cpu().float().item())
            if phase == 'validation':
                # Note that step should be called after validate()
                scheduler.step(epoch_loss)
                hist_train['val_loss'].append(epoch_loss.cpu().float().item())
                hist_train['val_acc'].append(epoch_acc.cpu().float().item())

            # Check if this is the best model wrt previous epochs
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                #best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(), PATH_WEIGHTS)
            
            if phase == "validation" and epoch_loss < best_loss:
                print('[INFO]: Val_loss improved from {} to {}'.format(best_loss, epoch_loss))
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(), PATH_MODEL)
                
                torch.save({'epoch': epoch, 
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss}, PATH_MODEL)
            

            if phase == "train" and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == "train" and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

            # Update learning rate
            #if phase == "train":
            #    scheduler.step()

            prefix = ''
            if phase == 'validation':
                prefix = 'val_'

            logs[prefix + 'loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
        
        if show_plot:
            liveloss.update(logs)
            liveloss.send()

    # Print final results and reload best weights
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    )
    print("Best test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))

    # Serializing json    
    with open(PATH_HISTORY, "w") as outfile:  
        json.dump(hist_train, outfile)  

    return model, hist_train