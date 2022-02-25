from model import *
import torch.utils.data
import torchvision.transforms as transforms
import argparse
import torch
import torchvision
import torch.nn as nn
import shutil
import os
device = torch.device("cuda" if use_cuda else "cpu")

class multiNet():

    def __init__(self, model_files):
        self.model_files = model_files
        self.models = []

    # DELETES OLD MODELS !
    def train_models(self, num=10, epochs=20, valid_size=1024, batch_size=32):
        if os.path.exists(self.model_files):
            shutil.rmtree(self.model_files)
        os.mkdir(self.model_files)
        print("Training " + str(num) + " models")
        train_transform = transforms.Compose([transforms.ToTensor()])
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)

        for i in range(num):
            net = Net()
            net.to(device)
            file_name = self.model_files + "/model_" + str(i) + ".pth"
            train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
            loss_net = train_model(net, train_loader, file_name, epochs)
            print("Training loss = ", loss_net)
            print("Model save to '{}'.".format(file_name))
        print("All models trained and saved to " + self.model_files)

    def load_models(self):
        for model in os.listdir(self.model_files):
            net = Net()
            net.load(self.model_files + '/' + model)
            self.models.append(net)

    # input.shape = (1,3,32,32)
    # label.shape = (1)
    # output.shape = (1,10)
    def prediction(self, images):
        outputs = torch.tensor([]).to(device)
        for model in self.models:
            model_pred = torch.max(model(images).data, 1)[1]
            outputs = torch.concat((outputs, model_pred))
        return torch.mode(outputs, 0)[0]

    def test_natural(self, test_loader):
        '''Basic testing function.'''

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                predicted = self.prediction(images)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def test_fgsm(self, test_loader, epsilon=0.2):
        correct = 0
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            images.requires_grad = True
            if self.prediction(images).item() != labels.item():
                continue
            gradients = []
            for model in self.models:
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction

                loss = criterion(outputs, labels)
                model.zero_grad()
                loss.backward()
                gradients.append(images.grad.data)

            data_grad = sum(gradients) / len(gradients)
            adv_data = fgsm(images, epsilon, data_grad)
            adv_pred = self.prediction(adv_data)
            if adv_pred.item() == labels.item():
                correct += 1
        final_acc = correct / float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
        return 100 * final_acc

    """def test_pgdlinf(self, test_loader, eta, eps, steps):
        correct = 0
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            adv_data = images.clone().detach().to(device)
            for _ in range(steps):
                # calculate outputs by running images through the network
                adv_data.requires_grad = True
                outputs = net(adv_data)

                _, nat_pred = torch.max(outputs.data, 1)
                if nat_pred.item() != labels.item():
                    break
                loss = criterion(outputs, labels)
                net.zero_grad()
                loss.backward()

                data_grad = adv_data.grad.data
                adv_data = pgdlinf(adv_data, images, eta, eps, data_grad)
            output = net(adv_data)
            _, adv_pred = torch.max(output.data, 1)
            if adv_pred.item() == labels.item():
                correct += 1
        final_acc = correct / float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(test_loader), final_acc))
        return 100 * final_acc"""


if __name__ == "__main__":
    m = multiNet("models/ensemble_models")
    m.train_models()
    m.load_models()
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, valid_size, 1)
    print(m.test_natural(valid_loader))
    print(m.test_fgsm(valid_loader))
