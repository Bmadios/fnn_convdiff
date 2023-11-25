import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from classic_fnn import *
import torch.optim as optim
import poutyne

# Supposons que 'data' soit un DataFrame contenant vos données
file_path = "solution_u_implicit_data.csv"
data = pd.read_csv(file_path)

# Fonctions de troncature
def truncate_two_decimals(value):
    return int(value * 100) / 100.0

def truncate_three_decimals(value):
    return int(value * 1000) / 1000.0

def is_multiple_of_10_minus_2(value, tolerance=1e-4):
    return abs(value * 100 - round(value * 100)) < tolerance

# Prétraitement des données
data["temps"] = data["temps"].apply(truncate_three_decimals)
data = data[data["temps"].apply(is_multiple_of_10_minus_2)]

data["x"] = data["x"].apply(truncate_two_decimals)
data["y"] = data["y"].apply(truncate_two_decimals)

# Séparation des entrées et des cibles
inputs = data.iloc[:, :-1].values  # Prend tout sauf la dernière colonne
targets = data.iloc[:, -1].values  # Prend seulement la dernière colonne

# Séparation en jeux de données d'entraînement, de validation et de test
seed = 42
X_train, X_temp, y_train, y_temp = train_test_split(inputs, targets, test_size=0.60, random_state=seed, shuffle=True)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, shuffle=True)

# Création des DataLoaders
train_loader = torch.utils.data.DataLoader(JeuDeDonnees(X_train, y_train), batch_size=1)
valid_loader = torch.utils.data.DataLoader(JeuDeDonnees(X_valid, y_valid), batch_size=1)
test_loader = torch.utils.data.DataLoader(JeuDeDonnees(X_test, y_test), batch_size=1)

# Initialisation du modèle
reseau = ClassicNeuralNetwork()
reseau.apply(init_poids_zero)

reseau.couche_entree.bias = torch.nn.Parameter(reseau.couche_entree.bias.to(torch.float64))
reseau.couche_entree.weight = torch.nn.Parameter(reseau.couche_entree.weight.to(torch.float64))

for i in range(3):
    reseau.couches_cachees[i].bias = torch.nn.Parameter(reseau.couches_cachees[i].bias.to(torch.float64))
    reseau.couches_cachees[i].weight = torch.nn.Parameter(reseau.couches_cachees[i].weight.to(torch.float64))

reseau.couche_sortie.bias = torch.nn.Parameter(reseau.couche_sortie.bias.to(torch.float64))
reseau.couche_sortie.weight = torch.nn.Parameter(reseau.couche_sortie.weight.to(torch.float64))

# Configuration de l'optimiseur et de la fonction de perte
optimizer = optim.Adam(reseau.parameters(), lr=0.0001)
fct_perte = torch.nn.MSELoss(reduction="mean")

# Entraînement du modèle
modele = poutyne.Model(reseau, optimizer, fct_perte, batch_metrics=["mse"])
_ = modele.fit_generator(train_generator=train_loader, valid_generator=valid_loader, epochs=500)

# Post-traitement et évaluation
unique_times = data['temps'].unique()
max_errors = []

for t in unique_times:
    if t >= 0.25 and t <= 0.58:
        indices = np.where(X_test[:, 0] == t)
        X_t = X_test[indices]
        Y_t = y_test[indices]

        X_tensor = torch.tensor(X_t, dtype=torch.float64)
        with torch.no_grad():
            predicted_u = reseau(X_tensor).numpy().squeeze()

        error = np.abs(Y_t - predicted_u)
        max_errors.append(np.max(error))

average_max_error = np.mean(max_errors)
FINAL_max_error = np.max(max_errors)

print(f"Moyenne des erreurs maximales sur les pas de temps: {average_max_error}")
print(f"MAXIMUM des erreurs maximales sur les pas de temps: {FINAL_max_error}")

# Affichage des pertes après l'entraînement
# (inclure le code de visualisation si nécessaire)
