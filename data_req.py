import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np


def load_data(path):
    print("Loading and preprocessing data from {} ...".format(path))
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
    }

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir= data_dir + '/valid'

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64) 
    }
    
    print("Loading of data finished...")
    return image_datasets['train'], dataloaders['train'], dataloaders['valid'], dataloaders['test']


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model. '''
    image = Image.open(image_path)
    
    # Resize and crop
    image = image.resize([256, 256]).crop((16, 16, 240, 240))
    
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = ((np_image - mean) / std)
    np_image = np_image.transpose((2, 0, 1))

    return np_image