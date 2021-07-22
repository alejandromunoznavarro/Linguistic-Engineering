# Practica 2 - Clasificacion de documentos
# Ingenieria Linguistica - MUIA 2020/21
#
# Clasificador documental
#
# Autores:
#   * Luna Jimenez Fernandez
#   * Alejandro Muñoz Navarro
#
# Este script se encarga de la clasificación de documentos en los tres temas elegidos
# (deporte, politica y salud)
#
# Para esto, el programa entrenara pares de metrica/modelos a partir de un conjunto de datos de entrenamiento,
# y posteriormente evaluara estos modelos con un conjunto de tests, para evaluar la tasa de acierto.
#
# Ademas de esto, el clasificador ordenara los documentos clasificados en cada tematica segun la probabilidad /
# confianza de que realmente pertenezcan a ese modelo.
#
# El objetivo es estudiar tanto la tasa de acierto del clasificador como la capacidad de ordenar los documentos.
#
# Las metricas utilizadas son las siguientes:
#   * Frecuencia absoluta
#   * TF-IDF
#
# Los modelos utilizados son los siguientes:
#   * kNN
#   * Naive Bayes
#   * SVM (Support Vector Machine)
#
# El conjunto de entrenamiento son los primeros 15 articulos de cada tema, mientras que
# el conjunto de test son los ultimos 15 articulos de cada tema
#
# Junto a los accuracies y el orden de los documentos clasificados,
# se devolvera una serie de graficas para facilitar el estudio de los resultados

###########
# IMPORTS #
###########

import os
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

#############
# GLOSARIOS #
#############

# Los glosarios se definen como listas para poder ser utilizados posteriormente
# por TF-IDF Vectorizer
sports_glossary = ["temporada", "temporadas",
                   "equipo", "equipos",
                   "nba",
                   "jugador", "jugadora", "jugadores", "jugadoras",
                   "millón", "millones",
                   "franquicia", "franquicias",
                   "contrato", "contratos",
                   "mercado", "mercados",
                   "partido", "partidos",
                   "juego", "juegos",
                   "historia", "historias",
                   "estrella", "estrellas",
                   "fútbol",
                   "olímpico", "olímpica", "olímpicos", "olímpicas",
                   "liga", "ligas",
                   "campeón", "campeona", "campeones", "campeonas",
                   "lesión", "lesiones",
                   "asamblea", "asambleas",
                   "nivel", "niveles",
                   "copa", "copas"]

politics_glossary = ["gobierno", "gobiernos",
                     "política", "políticas",
                     "presidente", "presidenta", "presidentes", "presidentas",
                     "rey", "reina", "reyes", "reinas",
                     "erc",
                     "españa",
                     "emérito", "emérita", "eméritos", "eméritas",
                     "portavoz", "portavoces",
                     "investigación", "investigaciones",
                     "euro", "euros",
                     "izquierda", "izquierdas",
                     "catalán", "catalana", "catalanes", "catalanas",
                     "tribunal", "tribunales",
                     "miembro", "miembros",
                     "agua", "aguas",
                     "director", "directora", "directores", "directoras",
                     "club", "clubes",
                     "exterior", "exteriores",
                     "grupo", "grupos",
                     "consejo", "consejos"]

health_glossary = ["caso", "casos",
                   "salud",
                   "paciente", "pacientes",
                   "pandemia", "pandemias",
                   "país", "países",
                   "virus", "viruses",
                   "persona", "personas",
                   "coronavirus",
                   "número", "números",
                   "dolor", "dolores",
                   "dato", "datos",
                   "cirugía", "cirugías",
                   "mundo", "mundos",
                   "situación", "situaciones",
                   "vida", "vidas",
                   "incidencia", "incidencias",
                   "enfermedad", "enfermedades",
                   "hospital", "hospitales",
                   "covid",
                   "vacuna", "vacunas"]


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

    Los textos se almacenan en un diccionario con la forma
    <nombre del tema (string)> => [lista de textos preprocesados (string)]

    A diferencia del extractor, en este caso los textos se almacenan reconstruidos (en vez de tokenizados)

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
            extracted_texts.append(extracted_text)

        # Con todos los ficheros extraidos, se almacenan en el diccionario
        print(topic.upper() + ": " + str(len(text_list)) + " ficheros leidos.")
        extracted_topics[topic] = extracted_texts

    return extracted_topics


def preprocess_text(text, stopwords_list):
    """
    Dado un texto, se preprocesa y se devuelve limpio.

    El tratamiento que se hace al texto es:
    - Se tokeniza el texto para trabajar con el

    - Todas las palabras se pasan a minusculas

      Las mayusculas son irrelevantes y solo pueden provocar terminos duplicados

    - Se eliminan los caracteres inutiles (puntos, interrogaciones, comillas...) y los numeros

      Estos caracteres no aportan informacion semantica relevante para el texto

    - Se eliminan las stopwords

      De nuevo, estas palabras no sirven para caracterizar los textos

    - Se reconstruye el texto a partir de los tokens

      En este caso si interesa reconstruir el texto, ya que es necesario para métricas como TF-IDF

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

    # Se reconstruye el texto, añadiendo un espacio entre tokens
    reunited_text = ' '.join(no_stopwords_tokens)

    return reunited_text


def extract_filenames(text_path):
    """
    Dada una ruta, extrae el la ruta de todos los ficheros / articulos usados.

    Usado principalmente para poder trabajar posteriormente con los documentos (copiandolos entre carpetas)

    Las rutas de los ficheros se almacenaran en un diccionario con la forma:
    <nombre del tema (string)> => [lista de textos preprocesados (string)]


    :param text_path: Ruta en la que se almacenan los textos
    :return: Dos diccionarios:
        - Diccionario conteniendo, para cada tema, los titulos de los ficheros
    """

    # Diccionario para almacenar los textos
    extracted_topics = {}

    # Muestra las carpetas (los temas) contenidos en el directorio
    topics = os.listdir(text_path)

    # Para cada topico, extrae su lista de nombres de ficheros
    for topic in topics:

        # Obten los nombres dentro de la carpeta del tema
        extracted_topics[topic] = [os.path.join(text_path, topic, filename) for filename in os.listdir(os.path.join(text_path, topic))]

    return extracted_topics


def execute_models(train_x, train_y, test_x, test_y, file_handle, filenames, metric_name):
    """
    Dado un conjunto de entrenamiento y un conjunto de test ya procesados,
    entrena los modelos con el conjunto de entrenamiento y estudia los resultados con el conjunto de test

    En concreto, los modelos entrenados son:
        * kNN
        * Naive Bayes
        * SVM (Maquina de Vector Soporte)

    Los parametros de algunos de estos modelos se optimizaran utilizando GridSearch para obtener
    la precisión máxima. Concretamente:
        * kNN =
            - Numero de vecinos (entre 1 y 10)
            - Medida para pesar las votaciones (uniforme, distancia euclidiana)
        * SVM =
            - Tipo de Kernel (lineal, polinomial, RBF (Radial Basis Function), logaritmico)
            - Para kernel polinomial, grado del polinomio (entre 2 y 5)

    Ademas, devuelve todos los accuracies en forma de diccionarios, de cara a
    calcular graficas posteriormente

    :param train_x: Caracteristicas del conjunto de entrenamiento
    :param train_y: Clasificaciones del conjunto de entrenamiento
    :param test_x: Caracteristicas del conjunto de test
    :param test_y: Clasificaciones del conjunto de test
    :param file_handle: Handle para el fichero donde se escribiran los resultados
    :param filenames: Lista con el nombre de los ficheros
    :param metric_name: Nombre de la metrica, usada para crear carpetas
    """

    # Imprime como titulo el nombre de la metrica
    print("- " + metric_name + " -\n")
    file_handle.write("- " + metric_name + " -\n\n")

    # kNN

    # Imprime titulo para el modelo
    print("* kNN:\n")
    file_handle.write("* kNN:\n\n")

    # Prepara el conjunto de parametros para kNN
    knn_grid = [{'n_neighbors': list(range(1, 11)), 'weights': ['uniform']},
                {'n_neighbors': list(range(1, 11)), 'weights': ['distance']}]

    # Crea el modelo de kNN usando GridSearch
    knn_classifier = GridSearchCV(
        KNeighborsClassifier(), knn_grid, scoring='accuracy'
    )

    # Ajusta el modelo
    knn_classifier.fit(train_x, train_y)

    # Imprime los parametros elegidos para el modelo
    print("Mejores hiperparametros:\n" +
          str(knn_classifier.best_params_) + "\n")
    file_handle.write("Mejores hiperparametros:\n" +
                      str(knn_classifier.best_params_) + "\n\n")

    # Obten el accuracy para el conjunto de test en general y para cada tematica
    knn_accuracy = knn_classifier.score(test_x, test_y)
    knn_outputs = knn_classifier.predict(test_x)
    knn_accuracy_sports, knn_accuracy_politics, knn_accuracy_health = compute_topics_accuracies(test_y, knn_outputs)

    # Imprime todas las accuracies
    print("Accuracy (general): " + "{0:.4f}".format(knn_accuracy) + "\n")
    file_handle.write("Accuracy (general): " + "{0:.4f}".format(knn_accuracy) + "\n\n")
    print("Accuracy (deportes): " + "{0:.4f}".format(knn_accuracy_sports))
    file_handle.write("Accuracy (deportes): " + "{0:.4f}".format(knn_accuracy_sports) + "\n")
    print("Accuracy (politica): " + "{0:.4f}".format(knn_accuracy_politics))
    file_handle.write("Accuracy (politica): " + "{0:.4f}".format(knn_accuracy_politics) + "\n")
    print("Accuracy (salud): " + "{0:.4f}".format(knn_accuracy_health))
    file_handle.write("Accuracy (salud): " + "{0:.4f}".format(knn_accuracy) + "\n")
    print("")
    file_handle.write("\n")

    # Ordena los documentos por probabilidad de ser elegidos
    knn_probabilities = knn_classifier.predict_proba(test_x)
    compute_documents_order(filenames, knn_probabilities, metric_name, "KNN", file_handle)

    # Naive Bayes

    # Imprime titulo para el modelo
    print("* Naive Bayes:\n")
    file_handle.write("* Naive Bayes:\n\n")

    # Ajusta el modelo
    nb_classifier = MultinomialNB()
    nb_classifier.fit(train_x, train_y)

    # Obten el accuracy para el conjunto de test en general y para cada tematica
    nb_accuracy = nb_classifier.score(test_x, test_y)
    nb_outputs = nb_classifier.predict(test_x)
    nb_accuracy_sports, nb_accuracy_politics, nb_accuracy_health = compute_topics_accuracies(test_y, nb_outputs)

    print("Accuracy (general): " + "{0:.4f}".format(nb_accuracy) + "\n")
    file_handle.write("Accuracy (general): " + "{0:.4f}".format(nb_accuracy) + "\n\n")
    print("Accuracy (deportes): " + "{0:.4f}".format(nb_accuracy_sports))
    file_handle.write("Accuracy (deportes): " + "{0:.4f}".format(nb_accuracy_sports) + "\n")
    print("Accuracy (politica): " + "{0:.4f}".format(nb_accuracy_politics))
    file_handle.write("Accuracy (politica): " + "{0:.4f}".format(nb_accuracy_politics) + "\n")
    print("Accuracy (salud): " + "{0:.4f}".format(nb_accuracy_health))
    file_handle.write("Accuracy (salud): " + "{0:.4f}".format(nb_accuracy) + "\n")
    print("")
    file_handle.write("\n")

    # Ordena los documentos por probabilidad de ser elegidos
    nb_probabilities = nb_classifier.predict_proba(test_x)
    compute_documents_order(filenames, nb_probabilities, metric_name, "NaiveBayes", file_handle)

    # SVM

    # Imprime titulo para el modelo
    print("* SVM (Maquina de Vector Soporte):\n")
    file_handle.write("* SVM (Maquina de Vector Soporte):\n\n")

    # Prepara el conjunto de parametros para kNN
    knn_grid = [{'kernel': ['linear']},
                {'kernel': ['poly'], 'degree': list(range(2,6))},
                {'kernel': ['rbf']},
                {'kernel': ['sigmoid']}]

    # Crea el modelo de kNN usando GridSearch
    svm_classifier = GridSearchCV(
        SVC(probability=True), knn_grid, scoring='accuracy'
    )

    # Ajusta el modelo
    svm_classifier.fit(train_x, train_y)

    # Imprime los parametros elegidos para el modelo
    print("Mejores hiperparametros:\n" +
          str(svm_classifier.best_params_) + "\n")
    file_handle.write("Mejores hiperparametros:\n" +
                      str(svm_classifier.best_params_) + "\n\n")

    # Obten el accuracy para el conjunto de test en general y para cada tematica
    svm_accuracy = svm_classifier.score(test_x, test_y)
    svm_outputs = svm_classifier.predict(test_x)
    svm_accuracy_sports, svm_accuracy_politics, svm_accuracy_health = compute_topics_accuracies(test_y, svm_outputs)

    # Imprime todas las accuracies
    print("Accuracy (general): " + "{0:.4f}".format(svm_accuracy) + "\n")
    file_handle.write("Accuracy (general): " + "{0:.4f}".format(svm_accuracy) + "\n\n")
    print("Accuracy (deportes): " + "{0:.4f}".format(svm_accuracy_sports))
    file_handle.write("Accuracy (deportes): " + "{0:.4f}".format(svm_accuracy_sports) + "\n")
    print("Accuracy (politica): " + "{0:.4f}".format(svm_accuracy_politics))
    file_handle.write("Accuracy (politica): " + "{0:.4f}".format(svm_accuracy_politics) + "\n")
    print("Accuracy (salud): " + "{0:.4f}".format(svm_accuracy_health))
    file_handle.write("Accuracy (salud): " + "{0:.4f}".format(svm_accuracy_health) + "\n")
    print("")
    file_handle.write("\n")

    # Ordena los documentos por probabilidad de ser elegidos
    svm_probabilities = svm_classifier.predict_proba(test_x)
    compute_documents_order(filenames, svm_probabilities, metric_name, "SVM", file_handle)

    # Devuelve, en diccionarios, todos los accuracies
    return {'knn': (knn_accuracy, knn_accuracy_sports, knn_accuracy_politics, knn_accuracy_health),
            'nb': (nb_accuracy, nb_accuracy_sports, nb_accuracy_politics, nb_accuracy_health),
            'svm': (svm_accuracy, svm_accuracy_sports, svm_accuracy_politics, svm_accuracy_health)}


def compute_topics_accuracies(test_y, predicted_y):
    """
    Dados los valores esperados para el conjunto de test y los valores realmente obtenidos,
    calcula manualmente el accuracy para cada tematica

    Las tematicas se clasifican como:
        * Deportes: 0
        * Politica: 1
        * Salud: 2

    :param test_y: Clasificaciones esperadas para el conjunto de test
    :param predicted_y: Clasificaciones obtenidas para el conjunto de test
    :return: Accuracy para deportes, politica y salud
    """

    # Indices de cada tematica
    sports_index = [i for i, x in enumerate(test_y) if x == 0]
    politics_index = [i for i, x in enumerate(test_y) if x == 1]
    health_index = [i for i, x in enumerate(test_y) if x == 2]

    # Extrae los elementos predichos para esos indices
    sports_predicted = [predicted_y[i] for i in sports_index]
    politics_predicted = [predicted_y[i] for i in politics_index]
    health_predicted = [predicted_y[i] for i in health_index]

    # Calcula las predicciones para cada tematica
    # Prediccion = (aciertos en esos indices) / (numero de indices)
    sports_accuracy = sports_predicted.count(0) / len(sports_index)
    politics_accuracy = politics_predicted.count(1) / len(politics_index)
    health_accuracy = health_predicted.count(2) / len(health_index)

    return sports_accuracy, politics_accuracy, health_accuracy


def compute_documents_order(document_paths, document_probabilities, metric_name, model_name, file_handle):
    """
    A partir de los nombres de los ficheros y de las probabilidades para cada fichero,
    crea una carpeta nueva donde se almacenan los ficheros ordenados por pertenencia a cada tematica

    Las tematicas son:
        * Deportes: 0
        * Politica: 1
        * Salud: 2

    :param document_paths: Ruta a cada fichero conteniendo un texto
    :param document_probabilities: Lista de probabilidades de pertenencia a cada clase de cada fichero
    :param metric_name: Nombre de la metrica usada
    :param model_name: Nombre del modelo usado
    :param file_handle: Handle del fichero, para escribir en el
    """

    # Nombre de la carpeta
    folder_name = "Clasificacion_" + metric_name + "_" + model_name

    # Comprueba si existe la carpeta previamente. Si no, la borra recursivamente
    if os.path.isdir(folder_name):
        shutil.rmtree(folder_name)

    # Crea la carpeta para almacenar los ficheros y el resto de carpetas internas para temas
    os.mkdir(folder_name)
    os.mkdir(os.path.join(folder_name, "Deporte"))
    os.mkdir(os.path.join(folder_name, "Politica"))
    os.mkdir(os.path.join(folder_name, "Salud"))

    # Obten la lista de documentos pertenecientes a cada tematica, incluyendo su "puntuacion"
    # Usando list comprehension

    # Deportes
    sports_documents = [(filename, probabilities[0]) for filename, probabilities in
                        zip(document_paths, document_probabilities) if np.argmax(probabilities) == 0]
    # Politica
    politics_documents = [(filename, probabilities[1]) for filename, probabilities in
                        zip(document_paths, document_probabilities) if np.argmax(probabilities) == 1]
    # Salud
    health_documents = [(filename, probabilities[2]) for filename, probabilities in
                        zip(document_paths, document_probabilities) if np.argmax(probabilities) == 2]

    # Ordena cada lista por el valor de su probabilidad
    # (de mayor a menor)
    sports_documents.sort(key=lambda tup: tup[1], reverse=True)
    politics_documents.sort(key=lambda tup: tup[1], reverse=True)
    health_documents.sort(key=lambda tup: tup[1], reverse=True)

    # Imprime titulo para la seccion
    print("= ORDENACION DE LOS DOCUMENTOS =\n")
    file_handle.write("= ORDENACION DE LOS DOCUMENTOS =\n\n")

    # Procesa cada lista

    # Deportes
    # Imprime titulo
    print("- Deportes")
    file_handle.write("- Deportes\n")

    for index, (filepath, score) in enumerate(sports_documents, 1):

        # Extrae el nombre del fichero
        filename = os.path.split(filepath)[1]

        # Imprime los valores en pantalla
        print(str(index) + "- " + filename + " (" + "{0:.4f}".format(score) + ")")
        file_handle.write(str(index) + "- " + filename + " (" + "{0:.4f}".format(score) + ")\n")

        # Copia el fichero a la carpeta
        shutil.copyfile(filepath, os.path.join(folder_name, "Deporte", "[" + str(index).zfill(2) + "] " + filename))

    print("")
    file_handle.write("\n")

    # Politica
    # Imprime titulo
    print("- Politica")
    file_handle.write("- Politica\n")

    for index, (filepath, score) in enumerate(politics_documents, 1):
        # Extrae el nombre del fichero
        filename = os.path.split(filepath)[1]

        # Imprime los valores en pantalla
        print(str(index) + "- " + filename + " (" + "{0:.4f}".format(score) + ")")
        file_handle.write(str(index) + "- " + filename + " (" + "{0:.4f}".format(score) + ")\n")

        # Copia el fichero a la carpeta
        shutil.copyfile(filepath, os.path.join(folder_name, "Politica", "[" + str(index).zfill(2) + "] " + filename))

    print("")
    file_handle.write("\n")

    # Salud
    # Imprime titulo
    print("- Salud")
    file_handle.write("- Salud\n")

    for index, (filepath, score) in enumerate(health_documents, 1):
        # Extrae el nombre del fichero
        filename = os.path.split(filepath)[1]

        # Imprime los valores en pantalla
        print(str(index) + "- " + filename + " (" + "{0:.4f}".format(score) + ")")
        file_handle.write(str(index) + "- " + filename + " (" + "{0:.4f}".format(score) + ")\n")

        # Copia el fichero a la carpeta
        shutil.copyfile(filepath, os.path.join(folder_name, "Salud", "[" + str(index).zfill(2) + "] " + filename))

    print("")
    file_handle.write("\n")


def print_graphs(freq_dict, tfidf_dict):
    """
    Dados todos los accuracies de todos los pares metrica-modelo en diccionarios,
    imprime graficas comparando las tasas de acierto de cada par metrica-modelo

    :param freq_dict: Diccionario conteniendo las tasas de acierto de Frecuencia absoluta
    :param tfidf_dict: Diccionario conteniendo las tasas de acierto de TF-IDF
    """

    # Extrae todas las tasas de acierto
    freq_knn_accuracy, freq_knn_accuracy_sports, freq_knn_accuracy_politics, freq_knn_accuracy_health = freq_dict['knn']
    freq_nb_accuracy, freq_nb_accuracy_sports, freq_nb_accuracy_politics, freq_nb_accuracy_health = freq_dict['nb']
    freq_svm_accuracy, freq_svm_accuracy_sports, freq_svm_accuracy_politics, freq_svm_accuracy_health = freq_dict['svm']

    tfidf_knn_accuracy, tfidf_knn_accuracy_sports, tfidf_knn_accuracy_politics, tfidf_knn_accuracy_health = tfidf_dict['knn']
    tfidf_nb_accuracy, tfidf_nb_accuracy_sports, tfidf_nb_accuracy_politics, tfidf_nb_accuracy_health = tfidf_dict['nb']
    tfidf_svm_accuracy, tfidf_svm_accuracy_sports, tfidf_svm_accuracy_politics, tfidf_svm_accuracy_health = tfidf_dict['svm']

    # Crea variables que van a ser reutilizadas por todos los graficos
    labels = ["Frecuencia absoluta", "TF-IDF"]
    x = np.arange(len(labels))
    bar_width = 0.15

    # Empieza a imprimir las graficas
    # GENERAL

    # Crea la grafica
    # Divide los datos
    knn_accuracy = [freq_knn_accuracy, tfidf_knn_accuracy]
    nb_accuracy = [freq_nb_accuracy, tfidf_nb_accuracy]
    svm_accuracy = [freq_svm_accuracy, tfidf_svm_accuracy]

    # Crea la grafica
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - bar_width, knn_accuracy, bar_width, label="KNN")
    ax.bar(x, nb_accuracy, bar_width, label="Naive Bayes")
    ax.bar(x + bar_width, svm_accuracy, bar_width, label="Maquina de Vector Soporte (SVM)")

    # Añade titulos, ejes y leyendas
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.linspace(0,1,11))
    ax.set_xticks(x),
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center')

    # Almacena la grafica en un fichero
    plt.tight_layout()
    plt.savefig('accuracy.png', bbox_inches='tight', dpi=600)
    plt.savefig('accuracy.eps', bbox_inches='tight', format='eps', dpi=600)

    # DEPORTES

    # Crea la grafica
    # Divide los datos
    knn_accuracy_sports = [freq_knn_accuracy_sports, tfidf_knn_accuracy_sports]
    nb_accuracy_sports = [freq_nb_accuracy_sports, tfidf_nb_accuracy_sports]
    svm_accuracy_sports = [freq_svm_accuracy_sports, tfidf_svm_accuracy_sports]

    # Crea la grafica
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_width, knn_accuracy_sports, bar_width, label="KNN")
    ax.bar(x, nb_accuracy_sports, bar_width, label="Naive Bayes")
    ax.bar(x + bar_width, svm_accuracy_sports, bar_width, label="Maquina de Vector Soporte (SVM)")

    # Añade titulos, ejes y leyendas
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(x),
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center')

    # Almacena la grafica en un fichero
    plt.tight_layout()
    plt.savefig('accuracy_sports.png', bbox_inches='tight', dpi=600)
    plt.savefig('accuracy_sports.eps', bbox_inches='tight', format='eps', dpi=600)


    # POLITICA

    # Crea la grafica
    # Divide los datos
    knn_accuracy_politics = [freq_knn_accuracy_politics, tfidf_knn_accuracy_politics]
    nb_accuracy_politics = [freq_nb_accuracy_politics, tfidf_nb_accuracy_politics]
    svm_accuracy_politics = [freq_svm_accuracy_politics, tfidf_svm_accuracy_politics]

    # Crea la grafica
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_width, knn_accuracy_politics, bar_width, label="KNN")
    ax.bar(x, nb_accuracy_politics, bar_width, label="Naive Bayes")
    ax.bar(x + bar_width, svm_accuracy_politics, bar_width, label="Maquina de Vector Soporte (SVM)")

    # Añade titulos, ejes y leyendas
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(x),
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center')

    # Almacena la grafica en un fichero
    plt.tight_layout()
    plt.savefig('accuracy_politics.png', bbox_inches='tight', dpi=600)
    plt.savefig('accuracy_politics.eps', bbox_inches='tight', format='eps', dpi=600)


    # SALUD

    # Crea la grafica
    # Divide los datos
    knn_accuracy_health = [freq_knn_accuracy_health, tfidf_knn_accuracy_health]
    nb_accuracy_health = [freq_nb_accuracy_health, tfidf_nb_accuracy_health]
    svm_accuracy_health = [freq_svm_accuracy_health, tfidf_svm_accuracy_health]

    # Crea la grafica
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_width, knn_accuracy_health, bar_width, label="KNN")
    ax.bar(x, nb_accuracy_health, bar_width, label="Naive Bayes")
    ax.bar(x + bar_width, svm_accuracy_health, bar_width, label="Maquina de Vector Soporte (SVM)")

    # Añade titulos, ejes y leyendas
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xticks(x),
    ax.set_xticklabels(labels)
    ax.legend(loc='upper center')

    # Almacena la grafica en un fichero
    plt.tight_layout()
    plt.savefig('accuracy_health.png', bbox_inches='tight', dpi=600)
    plt.savefig('accuracy_health.eps', bbox_inches='tight', format='eps', dpi=600)

    # Muestra todas las graficas
    plt.show()


####################
# CODIGO PRINCIPAL #
####################

# ANALISIS DE ENTRADA #

# Comprueba que existen al menos DOS argumento
# (la ruta a los ficheros de entrenamiento y la ruta a los ficheros de test)

if len(sys.argv) < 3:
    print("ERROR: Se esperaban al menos dos argumentos.\n"
          "USO ESPERADO: clasificador_documental <ruta a los articulos de entrenamiento> <ruta a los articulos de test>")
    sys.exit()

# Comprueba que los argumentos son rutas validas
if not os.path.isdir(sys.argv[1]):
    print("ERROR: El primer argumento debe ser una ruta a una carpeta valida.")
    sys.exit()

if not os.path.isdir(sys.argv[2]):
    print("ERROR: El segundo argumento debe ser una ruta a una carpeta valida.")
    sys.exit()

# Intenta cargar las stopwords
stopwords_list = load_stopwords()

# Teniendo ambos argumentos validos, extrae el texto de los ficheros en cada topico

# Textos de entrenamiento
training_texts_dict = extract_texts(sys.argv[1], stopwords_list)

# Textos de test
test_texts_dict = extract_texts(sys.argv[2], stopwords_list)

# Extrae ademas los nombres de fichero usados en el test
# Estos seran usados posteriormente para poder copiar y pegar los documentos de la carpeta original
# a la nueva carpeta (para ordenarlos adecuadamente)
test_filenames_dict = extract_filenames(sys.argv[2])

# Comprueba que se ha extraido al menos un fichero en cada conjunto
if not training_texts_dict:
    print("ERROR: La ruta especificada por el primer argumento no contiene carpetas o no sigue el formato esperado. "
          "(una carpeta con el nombre de cada tematica, y ficheros en formato txt en dicha carpeta)")
    sys.exit()

if not test_texts_dict:
    print("ERROR: La ruta especificada por el segundo argumento no contiene carpetas o no sigue el formato esperado. "
          "(una carpeta con el nombre de cada tematica, y ficheros en formato txt en dicha carpeta)")
    sys.exit()


# CÁLCULO DE METRICAS / CARACTERÍSTICAS DE LOS DOCUMENTOS #

# Genera una lista de los textos de entrenamiento y de test para ser procesada
training_texts = training_texts_dict["Deporte"] + training_texts_dict["Politica"] + training_texts_dict["Salud"]
test_texts = test_texts_dict["Deporte"] + test_texts_dict["Politica"] + test_texts_dict["Salud"]

# Genera una lista de los titulos de los ficheros
test_filenames = test_filenames_dict["Deporte"] + test_filenames_dict["Politica"] + test_filenames_dict["Salud"]

# Genera un glosario general a partir de los tres glosarios existentes
glossary = sports_glossary + politics_glossary + health_glossary

# Calcula las caracteristicas de cada documento usando cada una de las metricas

# Frecuencia absoluta
freq_model = CountVectorizer(vocabulary=glossary)
freq_train_features = freq_model.fit_transform(training_texts)
freq_test_features = freq_model.fit_transform(test_texts)

# TF-IDF
tfidf_model = TfidfVectorizer(vocabulary=glossary)
tfidf_train_features = tfidf_model.fit_transform(training_texts)
tfidf_test_features = tfidf_model.fit_transform(test_texts)


# CLASIFICACION #

# Genera las clases esperada para los textos de entrenamiento y de test
# Esto es facil de calcular, al estar ordenados sin mezclar
# Las clases son:
#   Deporte: 0
#   Politica: 1
#   Salud: 2

train_classes = ([0] * len(training_texts_dict["Deporte"])) + ([1] * len(training_texts_dict["Politica"])) + ([2] * len(training_texts_dict["Salud"]))
test_classes = ([0] * len(test_texts_dict["Deporte"])) + ([1] * len(test_texts_dict["Politica"])) + ([2] * len(test_texts_dict["Salud"]))

# Crea un fichero para ir almacenando los resultados para posterior estudio
file = open("clasificador_resultados.txt", "w", encoding="utf8")

# Imprime un titulo
print("-- RESULTADOS DEL CLASIFICADOR --\n")
file.write("-- RESULTADOS DEL CLASIFICADOR --\n\n")

# Obten los resultados para Frecuencia Absoluta
freq_dict = execute_models(freq_train_features, train_classes, freq_test_features, test_classes, file, test_filenames,"Frecuencia absoluta")

# Obten los resultados para TF-IDF
tfidf_dict = execute_models(tfidf_train_features, train_classes, tfidf_test_features, test_classes, file, test_filenames,"TF-IDF")

# Cierra el fichero para guardar los resultados
file.close()

# Imprime las graficas apropiadas
print_graphs(freq_dict, tfidf_dict)
