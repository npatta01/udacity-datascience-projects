import argparse
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


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(path, device='cpu'):
    logging.info(f"Loading {path}")
    #checkpoint = torch.load(path,map_location=device)
    checkpoint = torch.load(path)

    architecture = checkpoint['arch']
    pretrained_model = getattr(models, architecture)

    model = pretrained_model(pretrained = True)
    in_features = checkpoint['in_features']
    num_classes = len(checkpoint['class_to_idx'])
    
    hidden_units = checkpoint['hidden_units']
    
    classifier_layer = checkpoint['classifier_layer']
    
    additional_layers  =  torch.nn.Sequential(OrderedDict([
                          ('fc1', torch.nn.Linear(in_features, hidden_units)),
                          ('relu', torch.nn.ReLU()),
                          ('fc2', torch.nn.Linear(hidden_units, num_classes)),
                          ('output', torch.nn.LogSoftmax(dim=1))
                          ]))

    
    
    setattr(model, classifier_layer,additional_layers)


    model.load_state_dict(checkpoint['state'])
    model.class_to_idx = checkpoint['class_to_idx']


    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = PIL.Image.open(image)
    img_tensor = data_transforms["test"](img)
    
    num_dim = img_tensor.shape[0]
    if num_dim > 3:
        img_tensor = img_tensor[:3,::]
    
    return img_tensor.numpy()
    
    

def predict(image_path, model, cat_to_name,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.eval()
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
    
    # feed image to model
    logits = model(img_tensor)
    # get probabilities
    probs  = torch.exp(logits)
    # get top k
    top_probs, top_labels = probs.topk(topk)
    top_probs, top_labels = top_probs.data.tolist()[0]  , top_labels.data.tolist()[0]
    
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}

    top_labels = [idx_to_class[lab] for lab in top_labels]
    


   # flower names
    top_flowers = []
    
    for lab in top_labels:
        flower_name = cat_to_name[lab]
        top_flowers.append(flower_name)
    
    return top_probs, top_labels, top_flowers



parser = argparse.ArgumentParser(
    description='prediction using a neural network',
)

parser.add_argument('image', help="image to predict against")
parser.add_argument('checkpoint' , help="checkpoint of the model to use", default="checkpoints")
parser.add_argument('--category_names', help="mapping of categories to real names", default="cat_to_name.json")
parser.add_argument('--top_k', help="top KK most likely classes", default=3, type=int)
parser.add_argument('--gpu', help="predict using gpu", default=True, action='store_true')


"python predict.py assets/inference_example.png checkpoint --top_k 3 "

arguments = parser.parse_args()

print (arguments)

device = 'cpu'
if arguments.gpu:
    device = 'gpu'
    
    
with open(arguments.category_names, 'r') as f:
    cat_to_name = json.load(f)    

model = load_model(arguments.checkpoint, device)

predictions = predict(image_path=arguments.image, model=model, topk=arguments.top_k, cat_to_name=cat_to_name)

top_probs, top_labels, top_flowers = predictions

for tp,tl,tf in zip(top_probs, top_labels, top_flowers):
    logging.info({"prob":tp, "label": tl, "flowe_name":tf })

