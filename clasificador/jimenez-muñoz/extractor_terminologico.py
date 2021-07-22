# Practica 2 - Clasificacion de documentos
# Ingenieria Linguistica - MUIA 2020/21
#
# Extractor terminologico
#
# Autores:
#   * Luna Jimenez Fernandez
#   * Alejandro Muñoz Navarro
#
# Este script se encarga de la extraccion del glosario de los tres topicos
# usando pre-procesado de texto y frecuencia maxima
#
# Concretamente, el programa:
# - Leera los textos de cada tematica
# - Preprocesara los textos
# - Devolvera una lista con los 50 terminos de frecuencia maxima para cada tematica
#
# La lista devuelta sera procesada posteriormente de forma manual para obtener el glosario final

###########
# IMPORTS #
###########

import os
import sys

from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


######################
# METODOS AUXILIARES #
######################

def load_stopwords():
    """
    Carga las stopwords.

    Si hay un fichero "stopwords.txt" en el directorio, carga las palabras de ese fichero.
    En otro caso, carga las palabras de NLTK por defecto

    :return: Set con todas las stopwords
    """

    # Comprueba que existe el fichero
    if os.path.isfile("stopwords.txt"):

        # Existe el fichero: carga y devuelve el contenido del fichero
        # Se abre en UTF-8 para respetar acentos
        file = open("stopwords.txt", encoding="utf8")
        stopwords_list = set(file.read().split())

        # Cierra el fichero y devuelve las stopwords
        file.close()
        return stopwords_list

    else:

        # No existe el fichero: se cargan las stopwords de NLTK por defecto
        stopwords_list = set(stopwords.words("spanish"))
        return stopwords_list


def extract_texts(text_path, stopwords_list):
    """
    Dada una ruta, extrae todos los textos contenidos en su interior.
    Además de ser extraidos, los textos son preprocesados.

    Los textos se almacenan tokenizados (divididos palabra a palabra) en un diccionario con la forma
    <nombre del tema (string)> => [lista de palabras (string)]

    :param text_path: Ruta en la que se almacenan los textos
    :param stopwords_list: Lista conteniendo las stopwords a eliminar
    :return: Diccionario conteniendo, para cada tema, los las palabras en String
    """

    # Diccionario para almacenar los textos
    extracted_topics = {}

    # Muestra las carpetas (los temas) contenidos en el directorio
    topics = os.listdir(text_path)

    # Comprueba que se ha extraido al menos un topico
    if len(topics) == 0:
        # Si no, devuelve False (error)
        return False

    # Para cada topico, extrae su lista de textos
    for topic in topics:

        # Lista de textos extraidos
        extracted_texts = []

        # Obten los ficheros dentro de la carpeta del tema
        text_list = os.listdir(os.path.join(text_path, topic))

        # Para cada fichero, extrae el string
        for text in text_list:

            # Carga el fichero en UTF-8 (para compatibilidad con acentos)
            file = open(os.path.join(text_path, topic, text), encoding="utf8")

            # Convierte el fichero a string
            extracted_text = file.read()

            # Preprocesa el fichero
            extracted_text = preprocess_text(extracted_text, stopwords_list)

            # Cierra el fichero y almacenalo en la lista
            file.close()
            extracted_texts.extend(extracted_text)

        # Con todos los ficheros extraidos, se almacenan en el diccionario
        print(topic.upper() + ": " + str(len(text_list)) + " ficheros leidos.")
        extracted_topics[topic] = extracted_texts

    return extracted_topics


def preprocess_text(text, stopwords_list):
    """
    Dado un texto, se preprocesa el texto y se devuelve limpio.

    El tratamiento que se hace al texto es:
    - Se tokeniza el texto para trabajar con el

    - Todas las palabras se pasan a minusculas

      Las mayusculas son irrelevantes y solo pueden provocar terminos duplicados

    - Se eliminan los caracteres inutiles (puntos, interrogaciones, comillas...) y los numeros

      Estos caracteres no aportan informacion semantica relevante para el texto

    - Se eliminan las stopwords

      De nuevo, estas palabras no sirven para caracterizar los textos

    # En este caso no se reconstruira el texto original, ya que queremos contar la frecuencia absoluta de cada palabra.

    El objetivo de preprocesar el texto es reducir la carga de trabajo para el modelo, reduciendo la variabilidad.

    :param text: El texto sin tratar en forma de string
    :param stopwords_list: Lista conteniendo las stopwords a eliminar
    :return: Una lista de strings conteniendo el texto preprocesado
    """

    # Se tokeniza el texto (se extrae cada palabra)
    tokens = word_tokenize(text, language="spanish")

    # Se pasan las palabras a minusculas
    lowercase_tokens = [w.lower() for w in tokens]

    # Se eliminan los caracteres inutiles (caracteres no alfanumericos)
    stripped_tokens = [w for w in lowercase_tokens if w.isalpha()]

    # Se eliminan las stopwords
    no_stopwords_tokens = [w for w in stripped_tokens if w not in stopwords_list]

    return no_stopwords_tokens


####################
# CODIGO PRINCIPAL #
####################

# ANALISIS DE ENTRADA #

# Comprueba que existe al menos un argumento (con la ruta donde están contenidos los ficheros)

if len(sys.argv) < 2:
    print("ERROR: Se esperaba un argumento (ruta a la carpeta donde estan contenidos los ficheros para extraer el glosario)")
    sys.exit()

# Comprueba que el argumento es una ruta valida
if not os.path.isdir(sys.argv[1]):
    print("ERROR: El argumento debe ser una ruta válida a una carpeta.")
    sys.exit()

# Intenta cargar las stopwords
stopwords_list = load_stopwords()

# Teniendo el argumento valido, extrae el texto de los ficheros en cada topico
extracted_texts = extract_texts(sys.argv[1], stopwords_list)

# Comprueba que se ha extraido al menos un fichero
if not extracted_texts:
    print("ERROR: La ruta especificada por el argumento no contiene carpetas o no sigue el formato esperado. "
          "(una carpeta con el nombre de cada tematica, y ficheros en formato txt en dicha carpeta)")
    sys.exit()

# ESCRITURA DEL FICHERO #

# Prepara un fichero para almacenar toda la informacion
file = open("glosarios.txt", "w")

# Imprime un titulo para el glosario
print("\n - GLOSARIOS -\n")
file.write("- GLOSARIOS -\n")

# EXTRACCION DEL GLOSARIO INICIAL #

# Para cada tematica extraida, extrae los 50 terminos con mayor frecuencia
for topic in extracted_texts.keys():

    # Imprime un titulo para el tema
    print("\n" + topic.upper() + ":\n")
    file.write("\n" + topic.upper() + ":\n")

    # Crea una instancia de la clase Counter
    counted_words = Counter(extracted_texts[topic])

    # Muestra los 50 terminos mas frecuentes
    frequent_terms = counted_words.most_common(50)
    print(frequent_terms)

    # Escribe los terminos en el texto
    for term in frequent_terms:
        file.write(term[0] + " - (" + str(term[1]) + "), ")

# Cierra el fichero
file.close()
