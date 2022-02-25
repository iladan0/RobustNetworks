import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_arr, nbr_epochs):
  fig = plt.figure(figsize=(10,5))
  plt.plot(np.arange(1,nbr_epochs+1), loss_arr, "*-",label="Train_Loss")
  plt.title("Basic Network")
  plt.xlabel("Num of epochs")
  plt.xticks(np.arange(1, 20, 1))
  plt.legend()
  plt.show()

def plot_loss_dist(lossA_arr, lossF_arr, nbr_epochs):
  fig = plt.figure(figsize=(5,5))
  plt.plot(np.arange(1,nbr_epochs+1), lossA_arr, "*-",label="NetA_Loss")
  plt.plot(np.arange(1,nbr_epochs+1), lossF_arr, "o-",label="NetF_Loss")
  plt.title("NetworkA vs NetworkF loss")
  plt.xlabel("Num of epochs")
  plt.legend()
  plt.show()

def acc_vs_eps(attack_name, epsilons, accuracies):
  fig = plt.figure(figsize=(5,5))
  plt.plot(epsilons, accuracies, "*-")
  plt.title(attack_name)
  plt.ylim(bottom=0)
  plt.xlabel("Epsilon")
  plt.ylabel("Accuracy")
  plt.show()