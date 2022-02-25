from model import *
import torch.utils.data
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cpu")  # torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 32
criterion = nn.NLLLoss()


class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)


def get_noised_train_loader(noise, valid_size=valid_size, batch_size=batch_size):
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.Compose(
        [transforms.ToTensor(), AddGaussianNoise(0., noise)]))
    train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
    return train_loader


def get_noised_test_loader(noise, valid_size=valid_size):
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.Compose(
        [transforms.ToTensor(), AddGaussianNoise(0., noise)]))
    valid_loader = get_validation_loader(cifar, valid_size, 1)  # batch size one for attack test otherwise put 32
    return valid_loader


def train_network(noise, epochs=40):
    if not os.path.exists("models/random"):
        os.mkdir("models/random")
    net = Net()
    net.to(device)
    # TRAINING
    train_loader = get_noised_train_loader(noise=noise)
    model_file = 'models/random/randomized_' + str(noise) + '.pth'
    loss = train_model(net, train_loader, model_file, epochs)
    # VALIDATION
    acc, acc_noised = natural_acc(net, noise)
    print("Model trained with gaussian noise of {}, final loss of {}.\n "
          "Natural accuracy on classic test data is : {}, natural accuracy on noised test data is {}".format(noise,
                                                                                                             loss, acc,
                                                                                                             acc_noised))


def natural_acc(net, noise):
    valid_noised_loader = get_noised_test_loader(noise=noise)
    valid_loader = get_noised_test_loader(noise=0)

    acc = test_natural(net, valid_loader)
    acc_noised = test_natural(net, valid_noised_loader)
    return (acc, acc_noised)


def train_multiple(noises, epochs=40):
    for n in noises:
        train_network(n, epochs)


def get_training_noise(model_name):
    _, noise = model_name.split('_')
    noise = float(noise[:-4])
    return noise


def test_all_models(directory="models/random"):
    for model_name in os.listdir(directory):
        noise = get_training_noise(model_name)
        net = Net()
        net.load(directory + '/' + model_name)
        print("Testing model {} with noise {}".format(model_name, noise))
        acc, acc_noised = natural_acc(net, noise)
        print("Natural accuracy on testing data with no added noise : {}".format(acc))
        print("Natural accuracy on testing data with added {} noise : {}".format(noise, acc_noised))


def test_multiple_fgsm(model_name, noise):
    net = Net()
    net.load(model_name)
    epsilons = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    test_noised_loader = get_noised_test_loader(noise=noise)
    test_loader = get_noised_test_loader(noise=0)
    fgsm_noised_acc = []
    fgsm_acc = []
    print("Testing model : {}".format(model_name))
    for eps in epsilons:
        fgsm_noised_acc.append(test_fgsm(net, test_noised_loader, eps))
        fgsm_acc.append(test_fgsm(net, test_loader, eps))
    print("Epsilons : {}. Noise : {}".format(epsilons, noise))
    print("Accuracy /eps on noised testing data : {}".format(fgsm_noised_acc))
    print("Accuracy /eps on un noised testing data : {}".format(fgsm_acc))
    print("\n")


def test_multiple_noises(model_name, epsilon):
    noises = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    net = Net()
    net.load(model_name)
    fgsm_noised_acc = []
    for noise in noises :
        test_noised_loader = get_noised_test_loader(noise=noise)
        fgsm_noised_acc.append(test_fgsm(net, test_noised_loader, epsilon))
    print ("Testing model : {}".format(model_name))
    print("\n Noises : {} \n Epsilon : {}".format(noises, epsilon))
    print("Accuracy/test noise : {}".format(fgsm_noised_acc))


# on their own training noise
def fgsm_all_models(directory="models/random"):
    for model_name in os.listdir(directory):
        noise = get_training_noise(model_name)
        test_multiple_fgsm(directory + '/' + model_name, noise)


def attack_model(model_name, noise):
    net = Net()
    net.load(model_name)
    eta = 1 / 255
    steps = 40
    epsilon_arr = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    valid_noised_loader = get_noised_test_loader(noise=noise)
    print("FGSM attacks : ")
    fgsm_acc = []

    for eps in epsilon_arr:
        acc = test_fgsm(net, valid_noised_loader, eps)
        fgsm_acc.append(acc)
    print("PGD linf attacks : ")
    pgd_acc = []
    for eps in epsilon_arr:
        acc = test_pgdlinf(net, valid_noised_loader, eta, eps, steps)
        pgd_acc.append(acc)
    print("Accuracies with FGSM attack : {}".format(fgsm_acc))
    print("Accuracies with PGD attack : {}".format(pgd_acc))


if __name__ == "__main__":
    epsilon = 0.03
    model_name = "models/random/randomized_0.5.pth"
    test_multiple_noises(model_name, epsilon)
    model_name = "models/random/randomized_0.2.pth"
    test_multiple_noises(model_name, epsilon)
    model_name = "models/random/randomized_1.pth"
    test_multiple_noises(model_name, epsilon)

