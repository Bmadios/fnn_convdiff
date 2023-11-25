import torch
import poutyne
from poutyne import set_seeds
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from torch.optim import SGD
import numpy as np
import math
import torch.optim as optim

from sklearn.metrics import max_error, mean_absolute_error
from poutyne import SKLearnMetrics
my_epoch_metric = SKLearnMetrics([max_error, mean_absolute_error])

figure = {"dpi": 200, "figsize": [20, 12]}
plt.rc("figure", **figure)

seed = 42
class JeuDeDonnees(torch.utils.data.Dataset):
    def __init__(self, X, y):
        """
        Args:
                X: échantillons de données sous forme de ndarray de Numpy
                y: prédictions attendues associées aux échantillons présents dans X sous forme de ndarray de Numpy
        """
        super().__init__()

        self.echantillons = X
        self.predictions_attendues = y

    def __len__(self):
        return self.echantillons.shape[0]

    def __getitem__(self, idx):
        return self.echantillons[idx], self.predictions_attendues[idx]


# Réseau composé d'une couche d'entrée, trois couches cachées identiques et une couche de sortie
class ClassicNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.couche_entree = torch.nn.Linear(3, 64)
        self.couches_cachees = torch.nn.ModuleList([torch.nn.Linear(64, 64) for i in range(3)])
        self.tanh = torch.nn.Tanh()
        self.couche_sortie = torch.nn.Linear(64, 1)

    def forward(self, entree):
        temp = self.tanh(self.couche_entree(entree))
        for couche in self.couches_cachees:
            temp = self.tanh(couche(temp))

        sortie = self.couche_sortie(temp)

        return sortie

def init_poids_zero(m):
    """Fonction pour initialiser tous les poids et biais d'un réseau pleinement connecté à 0"""

    # On vérifie si le module est une couche pleinement connectée
    if isinstance(m, torch.nn.Linear):
        # Initialisation des poids
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu") ## Méthode He

        # Initialisation des biais
        torch.nn.init.zeros_(m.bias)