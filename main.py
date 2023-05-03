import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paso 1: cargar los datos y dividirlos en conjuntos de entrenamiento y prueba
datos = load_diabetes()
X = datos.data
y = datos.target
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=0)

# Paso 2: crear un objeto Ridge y entrenar el modelo
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_entrenamiento, y_entrenamiento)

# Paso 3: evaluar el modelo en el conjunto de prueba
y_prediccion = ridge_reg.predict(X_prueba)
mse = mean_squared_error(y_prueba, y_prediccion)
print('Error Cuadrático Medio:', mse)

# Paso 4: experimentar con diferentes valores de alpha
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_entrenamiento, y_entrenamiento)
    y_prediccion = ridge_reg.predict(X_prueba)
    mse = mean_squared_error(y_prueba, y_prediccion)
    print('Alpha:', alpha, 'Error Cuadrático Medio:', mse)