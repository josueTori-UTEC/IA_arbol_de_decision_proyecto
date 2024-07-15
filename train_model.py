# Librerias a usar
## Para trabajar con el dataframe
import pandas as pd
import numpy as np
## Para el modelo y visualizacion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

import joblib


# Alternativas del formulario y pesos de importancia que la IA debería aproximar

## Puntaje minimo para ser considerado organizado
organizacion_minima = 25

## el 0-2 de no prioridad a tener que priorizarlo
ciclo = {
    1:0,
    2:0,
    3:1,
    4:1,
    5:2,
    6:2,
    7:2,
    8:1,
    9:1,
    10:1
}

## el 1-3 a mayor dificultad
carrera = {
    "Administración y Negocios Digitales":1,
    "Bioingeniería":3,
    "Ciencia de la Computación":3,
    "Ciencia de Datos":2,
    "Ingeniería Ambiental":2,
    "Ingeniería Civil":3,
    "Ingeniería de la Energía":2,
    "Ingeniería Electrónica":2,
    "Ingeniería Industrial":1,
    "Ingeniería Mecánica":2,
    "Ingeniería Mecatrónica":3,
    "Ingeniería Química":3
}

## de 3 a 5 a mayor dificultad
num_cursos = {
    3:3,
    4:3,
    5:4,
    6:4,
    7:5,
    8:5
}


## de 2 a 10 en nivel de avance
h_estudio_sem = {
    20:10,
    15:8,
    10:6,
    5:4,
    0:2
}

## de 2 a 10 descendente
h_redes_dia = {
    8:2,
    6:4,
    4:6,
    2:8,
    0:10
}

## 2 a 10
tiempo_libre = {
    "Aprovecho en estudiar para mis cursos":10,
    "Repaso brevemente un curso":8,
    "Avanzo tareas cortas":6,
    "Veo mi celular para distraerme":4,
    "Salgo de fiesta":2
}

## 2 a 10
metodo_organizacion = {
    "Priorizar cursos de carrera":10,
    "Armar planes de estudio":8,
    "Coincidir con el horario de los amigos":4,
    "Meterse a la mayor cantidad de cursos posibles":2,
    "Elegir al azar los cursos":2
}


# Lista de opciones de respuestas
ciclo_keys = list(ciclo.keys())
carrera_keys = list(carrera.keys())
num_cursos_keys = list(num_cursos.keys())
h_estudio_sem_keys = list(h_estudio_sem.keys())
h_redes_dia_keys = list(h_redes_dia.keys())
tiempo_libre_keys = list(tiempo_libre.keys())
metodo_organizacion_keys = list(metodo_organizacion.keys())


# GENERANDO EL DUMMY DATA con 10 mil datos
dummy_data = {
    "ciclo": np.random.choice(ciclo_keys, 10000),
    "carrera": np.random.choice(carrera_keys,  10000),
    "num_cursos": np.random.choice(num_cursos_keys,  10000),
    "h_estudio_sem": np.random.choice(h_estudio_sem_keys,  10000),
    "h_redes_dia": np.random.choice(h_redes_dia_keys,  10000),
    "tiempo_libre": np.random.choice(tiempo_libre_keys,  10000),
    "metodo_organizacion": np.random.choice(metodo_organizacion_keys,  10000)
}
data = pd.DataFrame(dummy_data)


data_etiquetada = data.copy()
data_etiquetada['suma_fila'] = (data['ciclo'].map(ciclo) +
                   data['carrera'].map(carrera) +
                   data['num_cursos'].map(num_cursos) +
                   data['h_estudio_sem'].map(h_estudio_sem) +
                   data['h_redes_dia'].map(h_redes_dia) +
                   data['tiempo_libre'].map(tiempo_libre) +
                   data['metodo_organizacion'].map(metodo_organizacion))

data_etiquetada['organizacion'] = data_etiquetada['suma_fila'].apply(lambda x: 'bien organizado' if x > organizacion_minima else 'mal organizado')

# creamos df1 (el dataframe a usar en el entrenamiento)
df1 = data_etiquetada.copy()
df1 = df1.drop(columns=["suma_fila"])

# Convertimos Carrearas a un indice alfabeticamente

## Obtener las opciones únicas y ordenar alfabéticamente
opciones_unicas = sorted(df1['carrera'].unique())
## Crear un diccionario para mapear cada opción a un número del 1 al n
mapeo_opciones = {opcion: i + 1 for i, opcion in enumerate(opciones_unicas)}
## Aplicar la transformación a la columna 'Respuestas'
df1['carrera'] = df1['carrera'].map(mapeo_opciones)


# Convertimos lo que hacen en su tiempo libre a un indice alfabeticamente

## Obtener las opciones únicas y ordenar alfabéticamente
opciones_unicas = sorted(df1['tiempo_libre'].unique())
## Crear un diccionario para mapear cada opción a un número del 1 al n
mapeo_opciones = {opcion: i + 1 for i, opcion in enumerate(opciones_unicas)}
## Aplicar la transformación a la columna 'Respuestas'
df1['tiempo_libre'] = df1['tiempo_libre'].map(mapeo_opciones)


# Convertimos el metodo de organizacion a un indice alfabeticamente

## Obtener las opciones únicas y ordenar alfabéticamente
opciones_unicas = sorted(df1['metodo_organizacion'].unique())
## Crear un diccionario para mapear cada opción a un número del 1 al n
mapeo_opciones = {opcion: i + 1 for i, opcion in enumerate(opciones_unicas)}
## Aplicar la transformación a la columna 'Respuestas'
df1['metodo_organizacion'] = df1['metodo_organizacion'].map(mapeo_opciones)


# convertimos la variable de bien organizado o mal organizado a un valor binario (1: bien organizado, 0: mal organizado)
df1['organizacion'] = df1['organizacion'].map({'bien organizado': 1, 'mal organizado': 0})



# Separamos las variables independientes de la variable objetivo
X = df1.drop(columns=['organizacion'])
y = df1['organizacion']

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 200-> 94, 500->94.15
clf = RandomForestClassifier(n_estimators=100, random_state=52)
clf.fit(X_train, y_train)

joblib.dump(clf, 'model.pkl')