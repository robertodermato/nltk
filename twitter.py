import os
import nltk
from nltk.corpus import machado
from nltk.corpus import floresta
from nltk import ngrams, FreqDist
from nltk.metrics import *
import csv
import random
import math
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

tweets_negativos = []
tweets_positivos = []
tweets_geral = []
aval = []

with open('twits_data.csv', 'r', encoding='UTF-8') as file:
    reader = csv.reader(file, delimiter=";")
    for row in reader:
        if row[0][:1] == "1":
            tweets_positivos.append(row)
            tweets_geral.append(row)
            aval.append("positivo")
        elif row[0][:2] == "-1":
            tweets_negativos.append(row)
            tweets_geral.append(row)
            aval.append("negativo")
        else:
            continue

stopwords = nltk.corpus.stopwords.words('portuguese')
# print (stopwords)

# print(tweets_negativos)
# print(tweets_positivos)

stopwords.append("dell")
stopwords.append("notebook")
stopwords.append(",")
stopwords.append(".")
stopwords.append("!")
stopwords.append(":")
stopwords.append("?")
stopwords.append("note")
stopwords.append("#")
stopwords.append("@")
stopwords.append("...")
stopwords.append("-")
stopwords.append("http")
stopwords.append("pra")
stopwords.append("…")
stopwords.append("q")
stopwords.append("meu")
stopwords.append("not")

documents = []
for linha in tweets_negativos:
    frase = linha[1]
    palavras=[]
    tokens = nltk.word_tokenize(frase)
    for token in tokens:
        if token.lower() not in stopwords:
            palavras.append(token.lower())
    documents.append((palavras,"negativo"))
for linha in tweets_positivos:
    frase = linha[1]
    palavras=[]
    tokens = nltk.word_tokenize(frase)
    for token in tokens:
        if token.lower() not in stopwords:
            palavras.append(token.lower())
    documents.append((palavras,"positivo"))

# print(documents)

frases_negativas = []
frases_positivas = []

for linha in tweets_negativos:
    frases_negativas.append(linha[1])

for linha in tweets_positivos:
    frases_positivas.append(linha[1])

# print(frases_negativas)
# print(frases_positivas)

palavras_positivas = []
palavras_negativas = []

for frase in frases_negativas:
    tokens = nltk.word_tokenize(frase)
    for token in tokens:
        palavras_negativas.append(token.lower())

for frase in frases_positivas:
    tokens = nltk.word_tokenize(frase)
    for token in tokens:
        palavras_positivas.append(token.lower())

todas_palavras = palavras_negativas + palavras_positivas

todas_palavras_freq = nltk.FreqDist(todas_palavras)
palavras_features = list(todas_palavras_freq.keys())

# print(todas_palavras)
# print(palavras_negativas)
# print(palavras_positivas)

stemmer = nltk.stem.RSLPStemmer()

stems_negativos = []
stems_positivos = []

for palavra in palavras_negativas:
    if palavra not in stopwords:
        stem_negativo = stemmer.stem(palavra)
        stems_negativos.append(stem_negativo)

for palavra in palavras_positivas:
    if palavra not in stopwords:
        stem_positivo = stemmer.stem(palavra)
        stems_positivos.append(stem_positivo)

frequencia_de_dist_negativas = nltk.FreqDist(w.lower() for w in palavras_negativas if w not in stopwords)
frequencia_de_dist_positivas = nltk.FreqDist(w.lower() for w in palavras_positivas if w not in stopwords)

print("Palavra negativas mais comuns são ", frequencia_de_dist_negativas.most_common(10))
print("Palavra positivas mais comuns são ", frequencia_de_dist_positivas.most_common(10))

frequencia_de_dist_stems_negativas = nltk.FreqDist(w.lower() for w in stems_negativos if w not in stopwords)
frequencia_de_dist_stems_positivas = nltk.FreqDist(w.lower() for w in stems_positivos if w not in stopwords)

# print("Stems negativos mais comuns são ", frequencia_de_dist_stems_negativas.most_common(10))
# print("Stems positivos mais comuns são ", frequencia_de_dist_stems_positivas.most_common(10))

"""
tweets_geral_array = np.array(tweets_geral)
aval_array = np.array(aval)

vectorizer = CountVectorizer(analyzer = "word")
freq_tweets = vectorizer.fit_transform(tweets_geral_array)

modelo = MultinomialNB()
modelo.fit(freq_tweets, aval_array)
"""

def find_features(documento):
    words = set(documento)
    features = {}
    for w in palavras_features:
        features[w] = (w in words)
    return features

# print((find_features(todas_palavras)))

tagged = nltk.pos_tag(todas_palavras)
# print(tagged)

featuresets = [(find_features(msg), avaliacao) for (msg, avaliacao) in documents]
# print(featuresets)

porcentagem_para_treino = 0.8
porcentagem_para_teste = 1 - porcentagem_para_treino

# print ("Tweets positivos", len(tweets_positivos)) #197
# print ("Tweets negativos", len(tweets_negativos)) #407

tweets_positivos = random.shuffle(tweets_positivos)
random.shuffle(tweets_negativos)

quantidade_de_tweets_positivos_pro_treino = int(len(tweets_positivos)*0.8)
quantidade_de_tweets_negativos_pro_treino = int(len(tweets_negativos)*0.8)

# print (quantidade_de_tweets_positivos_pro_treino) #157
# print (quantidade_de_tweets_negativos_pro_treino) #325

set_treino_tweets_positivos = tweets_positivos[0:quantidade_de_tweets_positivos_pro_treino]
set_treino_tweets_negativos = tweets_negativos[0:quantidade_de_tweets_negativos_pro_treino]

# print (len(set_treino_tweets_positivos)) #157
# print (len(set_treino_tweets_negativos)) #325

set_teste_tweets_positivos = tweets_positivos[quantidade_de_tweets_positivos_pro_treino:]
set_teste_tweets_negativos = tweets_negativos[quantidade_de_tweets_negativos_pro_treino:]

# print (len(set_teste_tweets_positivos)) #40
# print (len(set_teste_tweets_negativos)) #82

print (set_teste_tweets_negativos)
print (set_teste_tweets_positivos)

# Calcula a distância eucliadiana entre um dado corrente (atual) e uma amostra
def euclidiana(atual, amostra):
      soma = 0;
      for i in len(atual):
          soma = soma + (atual[i] - amostra[i]) * (atual[i] - amostra[i])
      return math.sqrt(soma)

# x = stats.mode(speed) pra calcular moda

def ordena (distancia):
    aux = []
    for i in len(distancia):
        for j in len(distancia):
            if (distancia[j][0] > distancia[j + 1][0]):
                troca(distancia[j], distancia[j+1])


def troca (linha1, linha2):
    for i in len(linha1):
        aux = linha1[i]
        linha1[i] = linha2[i]
        linha2[i] = aux

