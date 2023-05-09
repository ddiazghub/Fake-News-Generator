from typing import TypeVar, Dict, List, Tuple
import nltk
import re
from nltk.corpus import stopwords
from nltk.util import ngrams
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request as request
import ssl
import json
from threading import Thread
from PIL import Image
from numpy import zeros

# Globales
STOP_WORDS = set(stopwords.words("spanish"))
plots = 0
context = ssl._create_unverified_context()

# Tipo genérico.
T = TypeVar('T')

"""
Se dividen las frases de una lista en palabras, se eliminan signos de puntuación.

Parámetros:
sentences - Lista de frases a separar en palabras.

Retorna:
Las frases divididas en palabras.
"""
def tokenize(sentences: List[str]) -> List[List[str]]:
    tokenized: List[str] = []

    # Iteramos a través de las oraciones.
    for sentence in sentences:
        # Separamos las oraciones en palabras y eliminamos los caracteres no alfanuméricos.
        tokens = [re.sub(r"\W+", "", token) for token in nltk.word_tokenize(sentence)]

        # Si una palabra está compuesta por caracteres exclusivamente no alfanuméricos, se elimina.
        tokenized.append([token for token in tokens if token != ""])
    
    return tokenized

"""
Se dividen las frases de una lista en palabras y se limpian, se eliminan signos de puntuación y stop words.

Parámetros:
sentences - Lista de frases a limpiar.

Retorna:
Las frases limpiadas.
"""
def tokenizeAndClean(sentences: List[str]) -> List[str]:
    newSentences: List[List[str]] = []

    # Iteramos a través de las oraciones.
    for sentence in sentences:
        # Separamos las oraciones en palabras y eliminamos los caracteres no alfanuméricos.
        tokens = [re.sub(r"\W+", "", token.lower()) for token in nltk.word_tokenize(sentence)]

        # Si una palabra está en la lista de stop words o está compuesta por caracteres exclusivamente no alfanuméricos, se elimina.
        newSentences.append([token for token in tokens if token != "" and not token.isnumeric() and token not in STOP_WORDS])

    return newSentences

"""
Se obtiene la DF de una lista de palabras.

Parámetros:
words - Lista de palabras.

Retorna:
Un diccionario donde las claves son las palabras y los valores son la frecuencia de cada palabra.
"""
def wordFrequencyDistribution(words: List[str]) -> Dict[str, int]:
    freqDist: Dict[str, int] = dict()

    # Iteramos a través de las palabras.
    for word in words:
        # Si la palabra no existe en el diccionario se añade con una frecuencia de 1, en caso contrario, se incrementa su frecuencia en 1.
        if word not in freqDist:
            freqDist[word] = 1
        else:
            freqDist[word] += 1

    return freqDist

"""
Se encuentra la palabra más frecuente en las noticias para cada día.

Parámetros:
news - Lista de artículos de BBC mundo, cada uno es una Tupla donde el primer elemento es la fecha y el segundo el titular.

Retorna:
Un diccionario donde las claves son los días y los valores son la palabra más frecuente en cada día.
"""
def mostCommonWordForDate(news: Tuple[str, List[str]]) -> Dict[str, str]:
    # Primero se genera una diccionario que tendrá las fechas como claves y la DF de palabras en esa fecha como valor.
    wordsForDates: Dict[str, Dict[str, int]] = dict()

    # Se itera a través de las noticias.
    for n in news:
        # Se extrae la fecha ignorando la hora.
        date = " ".join(n[0].split(" ")[1:])

        # Por cada palabra en el titular del artículo.
        for word in n[1]:
            # Si la fecha no existe en el diccionario se añade con un diccionario vacío como valor.
            if date not in wordsForDates:
                wordsForDates[date] = dict()

            # Si la palabra no existe en la DF de la fecha se añade con una frecuencia de 1, de lo contrario se incrementa su frecuencia.
            if word not in wordsForDates[date]:
                wordsForDates[date][word] = 1
            else:
                wordsForDates[date][word] += 1
    
    # Ahora se encuentra la palabra más frecuente para cada fecha.
    mostCommonForDate: Dict[str, str] = dict()

    # Se itera a través de cada clave y valor en el diccionario.
    for date, words in wordsForDates.items():
        # Se añade la fecha como clave y como valor la palabra con una mayor frecuencia en la DF de la fecha.
        mostCommonForDate[date] = max(words, key=words.get)

    return mostCommonForDate

"""
Se encuentra el número de artículos para cada día.

Parámetros:
news - Lista de artículos de BBC mundo, cada uno es una Tupla donde el primer elemento es la fecha y el segundo el titular.

Retorna:
Un diccionario donde las claves son los días y los valores son el número de artículo para ese día.
"""
def articlesForDate(news: List[Tuple[str, str]]) -> Dict[str, int]:
    artForDate: Dict[str, int] = dict()

    # Por cada noticia.
    for n in news:
        # Obtenemos fecha del artículo.
        date = " ".join(n[0].split(" ")[1:])

        # Se añade la fecha con una cuenta de 1 si no existe o se incrementa la cuenta en caso contrario.
        if date not in artForDate:
            artForDate[date] = 1
        else:
            artForDate[date] += 1
       
    return artForDate

"""
Se encuentra la DF de ngramas en las frases suministradas.

Parámetros:
sentences - Lista de frases de las cuales extraer ngramas.
n - Número de elementos en el ngrama.

Retorna:
Un diccionario donde las claves son los ngramas y los valores son la frecuencia de cada ngrama.
"""
def ngramFrequencyDistribution(sentences: List[List[str]], n: int) -> Dict[Tuple[str], int]:
    freqDist: Dict[Tuple[str], int] = dict()

    # Por cada frase.
    for sentence in sentences:
        # Se adquieren los ngramas en esa frase.
        grams = ngrams(sentence, n)
        
        # Por cada ngrama en la frase.
        for ngram in grams:
            # Se juntan en una sola string separada por espacios. (Inicialmente los ngramas son una tupla)
            ngram = " ".join(ngram)

            # Se añade el ngrama con una frecuencia de 1 si no existe o se incrementa la frecuencia en caso contrario.
            if ngram not in freqDist:
                freqDist[ngram] = 1
            else:
                freqDist[ngram] += 1

    return freqDist

"""
Se filtra una DF para obtener solamente los n valores más frecuentes.

Parámetros:
freqDist - Distribución de frecuencia a filtrar.
n - Número de elementos a obtener.

Retorna:
Una distribución de frencuencia que solo incluyo los n elmentos más frecuentes de la DF inicial.
"""
def getTopN(freqDist: Dict[T, int], n: int) -> Dict[T, int]:
    return dict(sorted(freqDist.items(), key=lambda item: item[1], reverse=True)[:n])

"""
Dibuja un gráfico de barras.

Parámetros:
data - Datos a graficar.
title - Título del gráfico.
xLabel - Etiqueta del eje X.
yLabel - Etiqueta del eje Y.
"""
def barGraph(data: Dict[T, int], title: str, xLabel: str, yLabel: str) -> None:
    fd = pd.DataFrame(data.items())
    global plots
    plt.figure(figsize=(13,6))
    plot = seaborn.barplot(x=0, y=1, data=fd)
    plot.set(xlabel = xLabel, ylabel=yLabel, title=title)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=20, fontsize=8)
    plots += 1
    plt.figure(plots)

"""
Dibuja una DF basada en un gráfico de barras.

Parámetros:
freqDist - Distribución de frecuencia.
title - Título del gráfico.
xLabel - Etiqueta del eje X.
"""
def plotFreqDist(freqDist: Dict[T, int], title: str, xLabel: str) -> None:
    barGraph(freqDist, title, xLabel, "Frecuencia")

"""
Dibuja una línea temporal.

Parámetros:
data - Datos a dibujar. (Por ahora solo acepta el diccionario de palabrás más frecuentes para cada fecha)
title - Título del gráfico.
"""
def timeline(data: Dict[str, str], title: str) -> None:
    length = len(data)

    # Se obtienen la fecha.
    dates = [key for key in reversed(data.keys())]
    global plots

    # Se inicializa el gráfico.
    plt.figure(figsize=(13, 4))
    axes = plt.gca()
    axes.set_ylim(-2, 1.75)

    # Se dibuja una línea horizontal para representar el eje del gráfico.
    axes.axhline(0, xmin=0.025, xmax=0.975, c="deeppink", zorder=1)

    # Se dibujan los puntos en los cuales se da un evento.
    axes.scatter(dates, zeros(length), s=120, c="palevioletred", zorder=2)
    axes.scatter(dates, zeros(length), s=30, c="darkmagenta", zorder=3)

    # Se añade la etiqueta para cada punto de tiempo.
    for i, date in enumerate(dates):
        axes.text(date, 0.4 if i % 2 == 0 else -0.8, f"{date}:\n{data[date]}", ha='center', fontfamily='serif', fontweight='bold', color='royalblue',fontsize=12)
    
    # Se añaden las líneas que conectan las etiquetas con sus puntos.
    stems = zeros(len(dates))
    stems[::2] = 0.3
    stems[1::2] = -0.3   
    markerline, stemline, baseline = axes.stem(dates, stems, use_line_collection=True)
    plt.setp(markerline, marker=',', color='darkmagenta')
    plt.setp(stemline, color='darkmagenta')
    plt.setp(axes.xaxis.get_majorticklabels(), rotation=20)

    # Se esconden Los ejes del gráfico.
    for spine in ["left", "top", "right", "bottom"]:
        axes.spines[spine].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])
 
    axes.set_title(title, fontweight="bold", fontfamily='serif', fontsize=16, color='royalblue')

    plots += 1
    plt.figure(plots)

"""
Dibuja una nube de palabras que se solicita de una API externa.

Parámetros:
words - Palabras que harán parte de esta nube de palabras, la frecuencia de estas afectará el tamaño de la letra.
"""
def wordcloud(words: List[str]):
    # La API se basa de una solo texto para dibujar la nube de palabras, se une la lista de palabras en una sola string separada por espacio.
    text = " ".join(words)

    # Para solicitar la nube de palabras a la API, se le debe enviar un objeto en formato JSON con la información de la solicitud. El texto se añade a una clave llamada 'text'.
    payload = json.dumps({
        "format": "png",
        "width": 1000,
        "height": 1000,
        "fontScale": 15,
        "scale": "linear",
        "removeStopwords": True,
        "minWordLength": 2,
        "text": text
    }).encode("utf-8")

    # Usamos urllib para realizar una solicitud POST a la API externa.
    req = request.Request("https://quickchart.io/wordcloud", method="POST")

    # Especificamos que el formato de los datos es JSON.
    req.add_header("Content-Type", "application/json")

    # Definimos un nuevo hilo en el cual se realizará la solicitud y se obtendrá la respuesta de manera asíncrona, de esta manera evitamos bloquear el resto del código mientras se espera a la respuesta de la API.
    def thread() -> None:
        # Se ejecuta la solicitud, asignando el objeto JSON como datos.
        with request.urlopen(req, data=payload, context=context) as response:
            # Una vez se obtiene la respuesta, leemos esta como una imagen.
            image = Image.open(response)
            image.show()

    # Se inicia el hilo.
    Thread(target=thread).start()



