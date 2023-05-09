import locale
import re
from typing import List, Tuple
from bs4 import BeautifulSoup
from datetime import date, datetime, timezone, timedelta
import pytz
import urllib.request as request

# Constantes globales
locale.setlocale(locale.LC_ALL, "es")
NEWS_URL = "https://www.bbc.com/mundo/topics/c2lej05epw5t"
today = date.today().strftime('%d %B %Y')

"""
Se obtienen n artículos de BBC Mundo mediante web-scrapping.

Parámetros:
n - Número de artículos a obtener

Retorna:
Una lista de tuplas donde cada tupla representa un artículo, siendo el primer elemento de la tupla la fecha de publicación y el segundo el titular.
"""
def scrap(n: int) -> List[Tuple[str, str]]:
    news = []

    # Iniciamos en la página de la sección de internacional.
    page = 1

    # Abrimos un archivo donde se guardarán las noticias.
    with open("news.csv", "w", encoding="utf-8") as file:
        # Repetimos hasta obtener n artículos
        while len(news) < n:
            # Se realiza una solicitud HTTP a la página de la sección internacional de BBC mundo
            with request.urlopen(f"{NEWS_URL}/page/{page}") as response:
                # Iniciamos la instancia de bs4 para analizar el html obtenido.
                soup = BeautifulSoup(response.read(), "html.parser")

                # Obtenemos todos los titulares y las fechas de publicación.
                timestamps = soup.find_all(attrs={ "class": "qa-post-auto-meta" })
                headlines = soup.find_all(id=re.compile("title_"))

                # Iteramos a través de los titulares y las fechas.
                for timestamp, headline in zip(timestamps, headlines):
                    # Formateamos la hora correctamente y la convertimos a la zona horaria local de Colombia. (Por defecto está en UTC)
                    ts = timestamp.contents[0]
                    ts = f"{ts} {today}" if len(ts) == 5 else f"0{ts} {today}" if len(ts) == 4 else ts
                    dt = datetime.strptime(ts, "%H:%M %d %B %Y").replace(tzinfo=timezone.utc)
                    dt = dt if dt < datetime.now(tz=timezone.utc) else dt - timedelta(days=1)
                    dt = dt.astimezone(pytz.timezone("America/Bogota"))
                    ts = dt.strftime("%H:%M %d %B %Y")
                    hl = headline.contents[0]

                    # Anñadimos la noticia a nuestra lista de noticias y la escribimos al archivo.
                    news.append((ts, hl))
                    file.write(f"{ts}\t{hl}\n")

            # Pasamos a la siguiente página de noticias.
            page += 1

    return news