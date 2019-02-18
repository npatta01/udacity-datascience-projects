# Imports here
import torch
from torchvision import datasets, transforms, models
import numpy as np
import time
import copy
import PIL
import json
import argparse
import logging
import os
import sys
import utility

from collections import OrderedDict

logging.basicConfig(stream=sys.stdout, level=logging.INFO)



data_transforms = utility.data_transforms


    
    
# reference: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model_helper(model, criterion, optimizer, scheduler, device, num_epochs,dataloaders,dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        logging.info("\n")

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train (data_dir, save_dir="checkpoints",arch="resnet152"
        ,learning_rate=0.01, use_gpu=False, epochs=10 , hidden_units=512):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train': datasets.ImageFolder(train_dir,data_transforms['train'])
                     ,'valid': datasets.ImageFolder(valid_dir,data_transforms['valid'])
                     ,'test': datasets.ImageFolder(valid_dir,data_transforms['test'])
                     }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train','valid','test']}

    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) 
                                  for x in ['train', 'valid','test']}
    
    
    
    ## INIT Network
    # Use GPU if it's available
    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    criterion = torch.nn.NLLLoss()
    
    classifier_layer = None
    classifier_layer_name = None
    try:
        classifier_layer =  model.fc
        classifier_layer_name ='fc'
    except:
        classifier_layer =  model.classifier
        classifier_layer_name ='classifier'

    
    
    num_ftrs = classifier_layer.in_features    
    additional_layers  =  torch.nn.Sequential(OrderedDict([
                          ('fc1', torch.nn.Linear(num_ftrs, hidden_units)),
                          ('relu', torch.nn.ReLU()),
                          ('fc2', torch.nn.Linear(hidden_units, len(class_names))),
                          ('output', torch.nn.LogSoftmax(dim=1))
                          ]))

    
    classifier_layer = additional_layers
    
    setattr(model, classifier_layer_name,additional_layers)

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = torch.optim.Adam(classifier_layer.parameters(), lr=learning_rate)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.to(device)
    
    logging.info (model)
    model = train_model_helper(model, criterion, optimizer, exp_lr_scheduler, device, epochs, dataloaders,dataset_sizes)
    
    
    os.makedirs(save_dir,exist_ok=True)
    
    torch.save(
    {
        'arch' : arch
        ,'state' : model.state_dict()
        ,'class_to_idx': image_datasets['train'].class_to_idx
        , 'in_features': num_ftrs
        , 'hidden_units': hidden_units
        , 'classifier_layer': classifier_layer_name

    }
    
    , f"{save_dir}/checkpoint.pth")
    

parser = argparse.ArgumentParser(
    description='training a neural network',
)

parser.add_argument('data_dir', help="location of the data")
parser.add_argument('--save_dir' , help="set directory to save checkpoints", default="checkpoints")
parser.add_argument('--arch', help="architecture of the neural network", default="resnet34")
parser.add_argument('--learning_rate', help="learning rate used during training", default=0.01, type=float)
parser.add_argument('--hidden_units', help="number of neural nets is hidden network", default=512, type=int)
parser.add_argument('--epochs', help="number of epochs to train for", default=5, type=int)
parser.add_argument('--gpu', help="train using gpu", default=True, action='store_true')



arguments = parser.parse_args()

"python train.py data"
logging.info (arguments)

train(data_dir=arguments.data_dir, save_dir=arguments.save_dir ,arch=arguments.arch
        ,learning_rate=arguments.learning_rate
        , use_gpu=arguments.gpu, epochs=arguments.epochs , hidden_units=arguments.hidden_units
     )