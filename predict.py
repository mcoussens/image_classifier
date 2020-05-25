import argparse
from torchvision import transforms, datasets, models, utils
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import OrderedDict
import json

parser = argparse.ArgumentParser(description='Train a neural network to classify images.')
parser.add_argument('path_to_image', help='Directory containing image data (str).', type=str)
parser.add_argument('checkpoint', help='Directory containing image data (str).', type=str)
parser.add_argument('--top_k', help='Directory containing image data (str).', type=int)
parser.add_argument('--category_names', help='Directory containing image data (str).', type=str)
parser.add_argument('--gpu', help='Directory containing image data (str).', type=str)
args = parser.parse_args()

img_path = args.path_to_image
checkpoint = args.checkpoint

# set k
if args.top_k:
    k = args.top_k
else:
    k = 1
    
# set category name mapping
if args.category_names:
    with open(args.category_names, 'r') as f:
        classes_map = json.load(f)
else:
    classes_map = None
    
# set device
if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
    
def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath)
    if checkpoint['device'] != device:
        return print('Please run program with same gpu/cpu that model was trained with.')
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True) 
    elif checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        return print('Model not compatible, please use vgg16 or vgg13')
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open (image_path)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    transform_image = image_transforms(image)
    return transform_image

def predict(image_path, model, k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    if device == 'cuda':
        image = image.type(torch.cuda.FloatTensor)
    else:
        image = image.type(torch.FloatTensor)
    prediction = torch.exp(model(image))
    probs, classes = prediction.topk(k)
    class_mapping = {result: flower for flower, result in model.class_to_idx.items()}
    probs = probs.cpu().numpy().tolist()[0]
    classes = classes.cpu().numpy().tolist()[0]
    classes = [class_mapping[result] for result in classes]
    return probs, classes

def map_classes(classes, class_map):
    if class_map == None:
        return classes
    else:
        mapped_classes = [class_map[str(result)] for result in classes]
    return mapped_classes

def prob_show(image_path, model, k, class_map, device):
    probs, classes = predict(img_path, model, k, device)
    names = map_classes(classes, class_map)
    for x in range(k):
        probability = probs[x]
        result = names[x]
        print('Rank: {}/{},  Class: {}, Probability: {}'.format(x+1, k, result, probability))
    return
    

model = load_checkpoint(filepath=checkpoint, device=device)
prob_show(img_path, model, k, classes_map, device)