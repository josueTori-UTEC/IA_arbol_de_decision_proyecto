from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

clf = joblib.load('model.pkl')

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

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form

    # Preprocesar los datos del formulario
    input_data = {
        "ciclo": ciclo[int(data['ciclo'])],
        "carrera": carrera[data['carrera']],
        "num_cursos": num_cursos[int(data['num_cursos'])],
        "h_estudio_sem": h_estudio_sem[int(data['h_estudio_sem'])],
        "h_redes_dia": h_redes_dia[int(data['h_redes_dia'])],
        "tiempo_libre": tiempo_libre[data['tiempo_libre']],
        "metodo_organizacion": metodo_organizacion[data['metodo_organizacion']]
    }
    
    df = pd.DataFrame([input_data])
    prediction = clf.predict(df)[0]

    return jsonify({'organizacion': 'bien organizado' if prediction == 1 else 'mal organizado'})

if __name__ == '__main__':
    app.run(debug=True)