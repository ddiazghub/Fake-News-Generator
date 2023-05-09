import nltk
nltk.download("punkt")
nltk.download("stopwords")
from word_chain_graph import WordChainGraph
from scrap_news import scrap
from analytics import articlesForDate, barGraph, getTopN, ngramFrequencyDistribution, plotFreqDist, timeline, wordFrequencyDistribution, mostCommonWordForDate, tokenizeAndClean, tokenize, wordcloud
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Se obtienen n noticias de BBC Mundo
    news = scrap(30)
    headlines = [n[1] for n in news]

    # Se limpian los titulares de las noticias para evitar stop words y puntuación
    cleaned = tokenizeAndClean(headlines)
    tokenized = tokenize(headlines)

    # Creamos un grafo de cadenas de Markov y le pasamos los títulos de las noticias limpiados
    graph = WordChainGraph()
    graph.load(tokenized)

    # Se extraen todas las palabras en los titulares de las noticias.
    words = [word for sentence in cleaned for word in sentence]

    # Dibujamos la nube de palabras con las palabras extraídas de las noticias.
    wordcloud(words)

    # Haciendo uso del grafo de cadenas de Markov, generamos n titulares de fake news, los cuales son escritos a un archivo llamado 'fakenews.csv'.
    with open("fakenews.csv", "w", encoding="utf-8") as file:
        for sentence in graph.generateSentences(30):
            file.write(sentence + "\n")

    # Calculamos estadísticas relevantes
    wfd = getTopN(wordFrequencyDistribution(words), 10) # 10 palabras mas comunes
    mcw = mostCommonWordForDate([(n[0], headline) for n, headline in zip(news, cleaned)]) # palabra más común por día.
    afd = articlesForDate(news) # Número de artículos por día.
    bigramfd = getTopN(ngramFrequencyDistribution(tokenized, 2), 10) # 10 bigramas mas comunes
    trigramfd = getTopN(ngramFrequencyDistribution(tokenized, 3), 10) # 10 trigramas mas comunes

    # Dibujamos gráficos
    plotFreqDist(wfd, "Distribución de frecuencia de palabras", "Palabras")
    plotFreqDist(bigramfd, "Distribución de frecuencia de bigramas", "Bigramas")
    plotFreqDist(trigramfd, "Distribución de frecuencia de trigramas", "Trigramas")
    timeline(mcw, "Palabra más frecuente por fecha")
    barGraph(afd, "Número de artículos por fecha", "Fecha", "Número de artículos")

    plt.show()
