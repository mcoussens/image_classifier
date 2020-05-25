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

parser = argparse.ArgumentParser(description='Train a neural network to classify images.')
parser.add_argument('data_directory', help='Directory containing image data (str).', type=str)
parser.add_argument('--save_dir', help='Directory to save model in (str).',type=str)
parser.add_argument('--arch', help='Neural network architecture ("vgg16" or "")', type=str)
parser.add_argument('--learning_rate', help='Set learning rate for model (float)', type=float)
parser.add_argument('--hidden_units', help='Set number of hidden units (float)', type=int)
parser.add_argument('--epochs', help='Set number of epochs (float)', type=int)
parser.add_argument('--gpu', help='Set whether to use GPU or not.', type=bool)
args = parser.parse_args()

# set save_directory
if args.save_dir:
    save_dir = args.save_dir
else:
    save_dir = None

# set learning rate 
if args.arch:
    if args.arch == 'vgg16':
        architecture = 'vgg16'
    elif args.arch == 'vgg13':
        architecture = 'vgg13'
    else:
        print('Model not supported')
else:
    architecture = 'vgg16'
  
# set learning_rate  
if args.learning_rate:
    lr = args.learning_rate
else:
    lr = 0.001
    
# set hidden units
if args.hidden_units:
    hidden_units = args.hidden_units
else:
    hidden_units = 500
    
# set epochs
if args.epochs:
    epochs = args.epochs
else:
    epochs = 3

# set device   
if args.gpu == True:
    device = 'cuda'
else:
    device = 'cpu'

def create_dataset(directory):
    train_dir = directory + '/train'
    test_dir = directory + '/valid'
    valid_dir = directory + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_datasets_train = datasets.ImageFolder(train_dir, transform=train_transforms) 
    classes = image_datasets_train.class_to_idx
    image_datasets_valid = datasets.ImageFolder(valid_dir, transform=test_transforms)                                         
    image_datasets_test = datasets.ImageFolder(test_dir, transform=test_transforms)

    dataloader_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=32, shuffle=True)                                       
    dataloader_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32, shuffle=True)                                       
    dataloader_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=32, shuffle=True)
    
    return dataloader_train, dataloader_valid, dataloader_test, classes

def load_model(architecture, device, hidden_units):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True) 
    elif architecture =='vgg13':
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 12544)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(12544, hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.to(device)
    return model

def train_model(model, dataloader_train, dataloader_valid, epochs, lr, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloader_train:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloader_valid:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloader_valid):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloader_valid):.3f}")
                running_loss = 0
                model.train()
    return model

def model_test(model, dataloader_test, device):
    accuracy = 0
    for inputs, labels in dataloader_test:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Calculate accuracy
        probs = torch.exp(outputs)
        top_p, top_class = probs.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test set accuracy: {accuracy/len(dataloader_test):.3f}")
    return

def save_model(model, save_dir, device, classes, architecture):
    if save_dir == None:
        return
    else:
        model.to(device)
        model.class_to_idx = classes

        checkpoint = {'device':device,
                    'architecture':architecture,
                    'state_dict': model.state_dict(),
                    'classifier': model.classifier,
                    'class_to_idx': model.class_to_idx}

        torch.save(checkpoint, save_dir)
        return print('Model saved')
    
dataloader_test, dataloader_train, dataloader_valid, classes = create_dataset(directory=args.data_directory)
loaded_model = load_model(architecture, device, hidden_units)
trained_model = train_model(loaded_model, dataloader_train, dataloader_valid, epochs, lr, device)
model_test(trained_model, dataloader_test, device)
save_model(trained_model, save_dir, device, classes, architecture)