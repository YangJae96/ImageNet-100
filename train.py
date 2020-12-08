import numpy as np
import time
import os
import cv2 as cv
import PIL
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision import models
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from config import cfg
from model import pyramid_net

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # Set it for torch to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"    # Multi-GPU Use

dtype = torch.float32

if torch.cuda.is_available():
    device=torch.device('cuda')

print('using device:', device)
print(torch.cuda.get_device_name())

simple_transform = transforms.Compose([  # Data preprocessing.
    transforms.Resize((224,224)),  # Resize it for imageNet-100 dataset
    transforms.ColorJitter(hue=.05, saturation=.05),
    transforms.RandomHorizontalFlip(p=0.7),   # Augmentation
    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR), # Augmentation
    transforms.ToTensor(),transforms.Normalize(cfg.pixel_mean, cfg.pixel_std) #Normalization
])
print("Loading ImageNet-100 Training Dataset.....", end=" ")

train=ImageFolder(cfg.train_dir, simple_transform) # Load the ImageNet-100 Dataset
train_data_gen = torch.utils.data.DataLoader(train,shuffle=True, batch_size=cfg.batch_size,
                                             num_workers=cfg.num_workers)
dataset_sizes = {'train':len(train_data_gen.dataset)}
dataloaders = {'train':train_data_gen}


def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(cfg.pixel_mean)
    std = np.array(cfg.pixel_std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    cv.imshow('test',inp)
    cv.waitKey(0)
    cv.destroyAllWindows()


def load_pretrained_model(model_name):
    """
        Load the pretrained model.
        Change the config.py for model_name.
        Use it for Transfer Learning(Fine tuning).
    """
    if model_name=="AlexNet":
        print("Loading pretrained AlexNet Model")
        model_ft = models.alexnet(pretrained=True)

        for param in model_ft.parameters():# Code for fixing the Conv Layer
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 100)
    elif model_name=="ResNet18":
        print("Loading pretrained ResNet18 Model")
        model_ft = models.resnet18(pretrained=True)

        for param in model_ft.parameters(): # Code for fixing the Conv Layer
            param.requires_grad = False # During Training Conv layer does not learn.
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 100)
    elif model_name=="ResNet50":
        print("Loading pretrained ResNet50 Model")

        model_ft = models.resnet50(pretrained=True)
        for param in model_ft.parameters():# Code for fixing the Conv Layer
            param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 100)
    elif model_name=="DenseNet":
        print("Loading pretrained DenseNet161 Model")
        model_ft = models.densenet161(pretrained=True)

        for param in model_ft.parameters():# Code for fixing the Conv Layer
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, 100)

    if cfg.load_model_true:
        model_ft.load_state_dict(torch.load(cfg.load_model_path))

    return model_ft

def load_custom_model(model_name):
    """
    Load the model.
    Change the config.py for model_name.
    """
    if  model_name=="AlexNet":
        print("Loading pretrained AlexNet Model")
        model = models.alexnet()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 100)
    elif model_name=="ResNet18":
        print("Loading ResNet18 Model")
        model = models.resnet18()  #Load the pytorch. torchvision model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100) #Set it to match the ImageNet-100 Classes.
    elif model_name=="ResNet50":
        print("Loading ResNet50 Model")
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100) #ImageNet-100 has 100 classes.
    elif model_name=="DenseNet":
        print("Loading DenseNet161 Model")
        model = models.densenet161()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 100)
    elif model_name=="MyNet":
        print("Loading Pyramid Model")
        model = pyramid_net.create_model()  # Load the model I implemented.

    if cfg.load_model_true: # Load the model that was stopped during training.
        model.load_state_dict(torch.load(cfg.load_model_path))

    return model



def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train']:
            
            model.train(True) #Set model to Train Mode
            running_loss = 0.0
            running_corrects=0
        
            for data in dataloaders[phase]:
                x, y = data
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                
                optimizer.zero_grad()
                scores = model(x)
                _, preds = torch.max(scores.data, 1)
                loss = criterion(scores, y)  # Calculate Loss: pred-target

                if phase =='train':
                    loss.backward()
                    optimizer.step()

                running_loss+=loss.data
                running_corrects+=torch.sum(preds==y.data)
            scheduler.step()
            epoch_loss = running_loss.item() / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train' and epoch_acc > best_acc:  # Save the best model
                    best_acc = epoch_acc
                    best_model_wts = model.module.state_dict()
                
        time_elapsed = time.time() - since
        print('Training on way in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
        if epoch % 10 ==0:
            print()
            print("Saving Model {} - Epoch".format(epoch))
            model_name = str(epoch)
            torch.save(best_model_wts, cfg.model_dir + '/' + model_name + "cvd.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Acc: {:4f}'.format(best_acc))
      
    return model

if __name__ == '__main__':

    if cfg.pretrained==1:
        model = load_pretrained_model(cfg.model_name)  # Call the Model
    else:
        model = load_custom_model(cfg.model_name)

    model = nn.DataParallel(model) # Set Model to GPU
    model.to(device=device)

    criterion = nn.CrossEntropyLoss() #set Loss Function
    optimizer_ft = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=cfg.lr_step_size,
                                           gamma=cfg.lr_dec_factor) # Scheduler

    model_trained = train_model(model, optimizer_ft, 
                       exp_lr_scheduler, num_epochs=cfg.epoch)  # Train Model with configs.

    torch.save(model_trained.module.state_dict(), cfg.result_dir + '/' + cfg.model_name+"_cvd.pth")
