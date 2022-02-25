#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from attacks import *
from utils import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 
criterion = nn.NLLLoss()

Temp = 100

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = F.log_softmax(x/Temp, dim=1)
        return x
    

    

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))



def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (TODO)'''
    print("Starting training")
    
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(),lr=0.0001, betas=(0.9, 0.999))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



    train_loss = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        loss_per_epoch = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = F.log_softmax(outputs/Temp,dim=1)
            
            loss = criterion(outputs, labels)
            loss_per_epoch+=loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = loss_per_epoch/len(train_loader)
        train_loss.append(epoch_loss)
        print(f'epoch : {epoch} --- loss : {epoch_loss} ')
        #scheduler.step()

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))
    return train_loss

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            
            outputs = net(images)
            outputs = F.log_softmax(outputs,dim=1)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
    
######################## Attack Testing ########################


def test_fgsm_D(net,test_loader,epsilon):
  correct = 0
  for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            images.requires_grad = True
            outputs = net(images)
            outputs = F.log_softmax(outputs,dim=1)
            # the class with the highest energy is what we choose as prediction
            _, nat_pred = torch.max(outputs.data, 1)
            if nat_pred.item() != labels.item():
              continue
            #if test sample is correctly predicted
            loss = criterion(outputs, labels)
            net.zero_grad()
            loss.backward()

            data_grad = images.grad.data
            adv_data = fgsm(images,epsilon,data_grad)
            output = net(adv_data)
            _,adv_pred = torch.max(output.data, 1)
            if adv_pred.item() == labels.item():
              correct+=1
  final_acc = correct/float(len(test_loader))
  print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
  return 100*final_acc


def test_pgd_D(net, test_loader, eta, eps, steps):
    correct = 0
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        adv_data = images.clone().detach().to(device)
        for _ in range(steps):
            # calculate outputs by running images through the network
            adv_data.requires_grad = True
            
            outputs = net(adv_data)
            outputs = F.log_softmax(outputs,dim=1)

            _, nat_pred = torch.max(outputs.data, 1)
            if nat_pred.item() != labels.item():
                break
            loss = criterion(outputs, labels)
            net.zero_grad()
            loss.backward()

            data_grad = adv_data.grad.data
            adv_data = pgd(adv_data, images, eta, eps, data_grad)
        output = net(adv_data)
        _, adv_pred = torch.max(output.data, 1)
        if adv_pred.item() == labels.item():
            correct += 1
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(test_loader), final_acc))
    return 100 * final_acc
##########################################

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    #### Create modelA and modelF 
    netA = Net()
    netA.to(device)

    netF = Net()
    netF.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model netA")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)

        loss_netA = train_model(netA, train_loader, args.model_file, args.num_epochs)
        print("Training loss netA = ",loss_netA)

        #Training netF on soft labels
        for data in train_loader:
            input, label  = data[0].to(device),data[1].to(device)
            softlabel  = F.log_softmax(netA(input),dim=1)
            data[1] = softlabel


        print("Training model netF")
        loss_netF = train_model(netF, train_loader, args.model_file, args.num_epochs)
        print("Training loss netF = ",loss_netF)
        print("Distilled Model save to '{}'.".format(args.model_file))

    
    

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    
    valid_loader = get_validation_loader(cifar, valid_size, 1) # batch size one for attack test otherwise put 32

    netF.load(args.model_file)

    acc = test_natural(netF, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

############ Testing attack #####################
    eta = 1 / 255
    steps = 40
    epsilon_arr = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2 , 0.3]
    fgsm_acc = []
    pgd_acc = []
    for eps in epsilon_arr:
      acc = test_fgsm_D(netF, valid_loader, eps)
      fgsm_acc.append(acc)
      print("Model fgsm attack accuracy (validation): {}".format(acc))
      acc = test_pgd_D(netF, valid_loader, eta, eps, steps)
      pgd_acc.append(acc)
      print("Model pgd attack accuracy (validation): {}".format(acc))
    
    print("FGSM accuracies = ",fgsm_acc)
    print("PGD accuracies = ",pgd_acc)
#################################################

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

