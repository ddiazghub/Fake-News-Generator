from typing import Dict, List
import random
import pprint

"""
Clase que representa una cadena de Markov que se implementa mediante un grafo dirigido que al mismo tiempo se implementa con un diccionario anidado.

Parámetros:
graph - Diccionario que representa al grafo, en este se guarda cada palabra como clave y un diccionario que contiene las transiciones de la forma [palabra -> peso].
El peso de cada transición el número de veces que se esta se da, por lo cual la probabilidad de cada transición es peso / (# total de transiciones en la palabra).
"""
class WordChainGraph:
    # Atributos estáticos que contienen las palabras que representan el inicio y el fin de una oración.
    START = "\0|"
    END = "|\0"

    __graph: Dict[str, Dict[str, int]]

    def __init__(self) -> None:
        self.__graph = dict()

    """
    Carga un conjunto de frases a la cadena de Markov para que se generen frases a partir de estas.
    
    Parámetros:
    sentences - Frases para insertar en el grafo
    """
    def load(self, sentences: List[str]):
        # Se reinicia el grafo.
        self.__graph.clear()
        self.__graph[WordChainGraph.START] = dict()
        self.__graph[WordChainGraph.END] = dict()
        
        # Iteramos a través de las frases.
        for sentence in sentences:
            # Iteramos a través de cada palabra en la frase actual.
            for i in range(len(sentence)):
                # Si la palabra no se encuentra como clave en el diccionario, la añadimos con un diccionario vacío como valor.
                if sentence[i] not in self.__graph:
                    self.__graph[sentence[i]] = dict()

                # Si la palabra es la primera palabra de la frase, la añadimos como una transición de la palabra de inicio.
                if i == 0:
                    if sentence[i] not in self.__graph[WordChainGraph.START]:
                        self.__graph[WordChainGraph.START][sentence[i]] = 1
                    else:
                        self.__graph[WordChainGraph.START][sentence[i]] += 1
                
                # Si la palabra está al final de la oración, añadimos a la palabra de fin como una transición de esta palabra, de lo contrario añadimos a la siguiente palabra en la frase como transición.
                chain = WordChainGraph.END if i == len(sentence) - 1 else sentence[i + 1]

                if chain not in self.__graph[sentence[i]]:
                    self.__graph[sentence[i]][chain] = 1
                else:
                    self.__graph[sentence[i]][chain] += 1

    """
    Se generan n frases aleatorias haciendo uso de la información en el grafo.
    
    Parámetros:
    n - Número de frases a generar.

    Retorna:
    Una lista de frases generadas de manera aleatoria.
    """
    def generateSentences(self, n: str = 10) -> List[str]:
        sentences: List[str] = []
        
        # Generamos n frases.
        for i in range(n):
            sentence = ""

            # Iniciamos en la palabra de inicio y realizamos una transición aleatoria teniendo en cuenta los pesos de cada transición. Esta transición nos va a retornar la palabra con la cual iniciaremos nuestra frase.
            chains = self.__graph[WordChainGraph.START].items()
            current = random.choices([chain[0] for chain in chains], weights=[chain[1] for chain in chains])[0]

            # Realizamos transiciones hasta que se llegue a la palabra de fin.
            while current != WordChainGraph.END:
                # Añadimos la palabra actual a la frase.
                sentence += current + " "

                # Obtenemos las transiciones posibles desde la palabra actual.
                chains = self.__graph[current].items()
                
                # Realizamos una transición aleatoria teniendo en cuenta los pesos de cada transición. Esto nos retorna otra palabra.
                current = random.choices([chain[0] for chain in chains], weights=[chain[1] for chain in chains])[0]

            # Añadimos la frase generada a nuestra lista de frases.
            sentences.append(sentence[:-1] + ".")

        return sentences
            

