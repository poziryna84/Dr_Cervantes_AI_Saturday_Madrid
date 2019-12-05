import os
from flask import Flask, request, render_template
from sklearn.externals import joblib
from unicodedata import normalize
from nltk.corpus import stopwords
import gensim
import xml.etree.cElementTree as ET
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from unicodedata import normalize
from bs4 import BeautifulSoup

app: Flask = Flask(__name__)

modelo = None


def cleanText(text):
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r' ', text)
    text = text.lower()
    text = text.replace('\n', '')  # Para quitar los \n
    text = re.sub(r'\d*\/\d+\/\d*', r'', text)
    text = re.sub(r'(\\)', r'', text)  # Para quitar \\\\\\
    text = re.sub(r'[-,:,!,¡,?,¿,_,/,...,+,),(,*]', r' ', text)  # Para quitar muchos caracteres
    text = re.sub(r'(\s){2,}', r' ', text)  # Para quitar muchos espacios
    text = re.sub(r'\d*', r'', text)  # Para quitar los \n
    text = re.sub(r'(\s){2,}', r' ', text)  # Para quitar muchos espacios
    # -> NFD y eliminar diacríticos (quitar acentos y demás...)
    text = re.sub(
        r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",
        normalize("NFD", text), 0, re.I
    )
    # -> NFC
    text = normalize('NFC', text)
    return text


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue

            # Añadimos a las stopwords de NLTK las propias de Sanidad
            sw_all = stopwords.words('spanish')
            sw_all += ['enfermedad', 'enfermedades', 'desorden', 'síntoma', 'síntomas', 'drogas', 'drogas', 'problemas',
                       'antecedente', 'antecedentes', 'medicamentos', 'medicamento', 'píldora', 'píldoras',
                       'tratamiento', 'tratamientos', 'cápsula', 'tableta', 'tabletas', 'pestañas', 'médico', 'dr',
                       'dra', 'doc', 'médicos', 'prueba', 'pruebas', 'especialista', 'especialistas',
                       'efecto secundario', 'efectos secundarios', 'farmacéutico', 'farmacéutico', 'farmacéutico',
                       'diagnóstico', 'diagnóstico', 'diagnosticado', 'examen', 'desafío', 'dispositivo', 'condición',
                       'condiciones', 'sufrimientos', 'sensación', 'sensacion', 'prescripción', 'prescribir',
                       'administrar', 'administración', 'prescrito', 'receta', 'paciente', 'pacientes', 'antecedente',
                       'antecedentes', 'síntoma', 'sindrome', 'medicina', 'medicinas', 'test', 'prueba', 'diagnóstica',
                       'efectos', 'secundarios', 'secundario', 'ingreso', 'ingresos', 'diferenciado', 'semana',
                       'semanas', 'mes', 'meses', 'dia', 'dias', 'paciente', 'pacientes']

            if word in sw_all:
                continue

            tokens.append(word.lower())

    return tokens


methods = ['GET', 'POST']
GET, POST = methods


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route('/consultar', methods=methods)
def consultar():
    problema = request.form['problema']

    global model_dbow
    global logreg
    global clf
    global resultado

    if (os.path.isfile("models/model_dbow_D2V_v8.joblib")):
        model_dbow = joblib.load("models/model_dbow_D2V_v8.joblib")
    logreg = joblib.load('models/logreg_D2V_v8.joblib')
    clf = joblib.load('models/clf_D2V_v8.joblib')
    print("Modelo cargado")

    # user_query = 'Desde anoche refiere tos intensa y ahogos'
    user_query = problema

    user_query = cleanText(user_query)  # limpiamos
    user_query = (tokenize_text(user_query))  # tokenizamos
    user_query_vectorized = model_dbow.infer_vector(user_query)
    docu = []
    docu.append(user_query_vectorized)
    docu = tuple(docu)
    grupo =  logreg.predict(docu)

    if grupo == 1:
        categoria = "ENFERMEDADES INFECCIOSAS Y PARASITARIAS (001-139)"
    elif grupo == 2:
        categoria = "NEOPLASIAS (140-239)"
    elif grupo == 3:
        categoria =  "ENFERMEDADES ENDOCRINAS, DE LA NUTRICION Y METABOLICAS Y TRASTORNOS DE LA INMUNIDAD (240-279)"
    elif grupo == 4:
        categoria =  "ENFERMEDADES DE LA SANGRE Y DE LOS ORGANOS HEMATOPOYÉTICOS (280-289)"
    elif grupo == 5:
        categoria =  "TRASTORNOS MENTALES, DEL COMPORTAMIENTO Y EL DESARROLLO NEUROLÓGICO (290-319)"
    elif grupo == 6:
        categoria =  "ENFERMEDADES DEL SISTEMA NERVIOSO Y DE LOS ÓRGANOS DE LOS SENTIDOS (320-389)"
    elif grupo == 7:
        categoria =  "ENFERMEDADES DEL SISTEMA CIRCULATORIO (390-459)"
    elif grupo == 8:
        categoria =  "ENFERMEDADES DEL APARATO RESPIRATORIO (460-519)"
    elif grupo == 9:
        categoria =  "ENFERMEDADES DEL APARATO DIGESTIVO (520-579)"
    elif grupo == 10:
        categoria =  "ENFERMEDADES DEL APARATO GENITOURINARIO (580-629)"
    elif grupo == 11:
        categoria =  "COMPLICACIONES DEL EMBARAZO, PARTO Y PUERPERIO (630-679)"
    elif grupo == 12:
        categoria =  "ENFERMEDADES DE LA PIEL Y DEL TEJIDO SUBCUTÁNEO (680-709)"
    elif grupo == 13:
        categoria =  "ENFERMEDADES DEL SISTEMA OSTEO-MIOARTICULAR Y TEJIDO CONJUNTIVO (710-739)"
    elif grupo == 14:
        categoria =  "ANOMALIAS CONGÉNITAS (740-759)"
    elif grupo == 15:
        categoria =  "CIERTAS ENFERMEDADES CON ORIGEN EN EL PERÍODO PERINATAL (760-779)"
    elif grupo == 16:
        categoria =  "SÍNTOMAS, SIGNOS Y ESTADOS MAL DEFINIDOS (780-799)"
    elif grupo == 17:
        categoria =  "LESIONES Y ENVENENAMIENTOS (800-999)"
    else:
        categoria =  "OTROS."

    print('Categoría: ')
    print(categoria)

    return render_template("index.html", problema=problema, solucion=categoria)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80, debug=True)
