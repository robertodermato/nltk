import nltk
import csv
import random
import math
from scipy import stats

# 90 na bag e 25k dá bom
qtidade_palavras_bag_of_words = 90
k_vizinhos = 25

tweets_negativos = []
tweets_positivos = []

with open('twits_data.csv', 'r', encoding='UTF-8') as file:
    reader = csv.reader(file, delimiter=";")
    for row in reader:
        if row[0][:1] == "1":
            tweets_positivos.append(row)
        elif row[0][:2] == "-1":
            tweets_negativos.append(row)
        else:
            continue

# print(tweets_negativos)
# print(tweets_positivos)
# tweets_negativos.append(['-1','Que bela porcaria de computador!'])

stopwords = nltk.corpus.stopwords.words('portuguese')
# print (stopwords)

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

porcentagem_para_treino = 0.8
porcentagem_para_teste = 1 - porcentagem_para_treino
# print ("Tweets positivos", len(tweets_positivos)) #197
# print ("Tweets negativos", len(tweets_negativos)) #407

# random.shuffle(tweets_positivos)
# random.shuffle(tweets_negativos)

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

bag_of_words=[]
aval_pos_bow = []
aval_neg_bow = []
for linha in set_treino_tweets_negativos:
    tweet = linha[1]
    tokens = nltk.word_tokenize(tweet)
    for token in tokens:
        if token.lower() not in stopwords:
            bag_of_words.append(token.lower())
            aval_neg_bow.append(token.lower())
for linha in set_treino_tweets_positivos:
    tweet = linha[1]
    tokens = nltk.word_tokenize(tweet)
    for token in tokens:
        if token.lower() not in stopwords:
            bag_of_words.append(token.lower())
            aval_pos_bow.append(token.lower())

# print(bag_of_words)

stemmer = nltk.stem.RSLPStemmer()

stemmed_bag_of_words = []
pos_stemmed_bag_of_words = []
neg_stemmed_bag_of_words = []

for word in bag_of_words:
    stemmed_bag_of_words.append(stemmer.stem(word))

for word in aval_pos_bow:
    pos_stemmed_bag_of_words.append(stemmer.stem(word))

for word in aval_neg_bow:
    neg_stemmed_bag_of_words.append(stemmer.stem(word))


# print (stemmed_bag_of_words)
# print (pos_stemmed_bag_of_words)
# print (neg_stemmed_bag_of_words)

frequencia_de_dist = nltk.FreqDist(w.lower() for w in stemmed_bag_of_words)
frequencia_de_dist_pos = nltk.FreqDist(w.lower() for w in pos_stemmed_bag_of_words)
frequencia_de_dist_neg = nltk.FreqDist(w.lower() for w in neg_stemmed_bag_of_words)
# print("Palavras mais comuns são ", frequencia_de_dist.most_common(50))

sorted_stemmed_bag_of_words = dict(sorted(frequencia_de_dist.items(), key=lambda x: x[1], reverse=True))
sorted_pos_stemmed_bag_of_words = dict(sorted(frequencia_de_dist_pos.items(), key=lambda x: x[1], reverse=True))
sorted_neg_stemmed_bag_of_words = dict(sorted(frequencia_de_dist_neg.items(), key=lambda x: x[1], reverse=True))
keys_sorted_stemmed_bag_of_words = list(sorted_stemmed_bag_of_words.keys())
keys_sorted_pos_stemmed_bag_of_words = list(sorted_pos_stemmed_bag_of_words.keys())
keys_sorted_neg_stemmed_bag_of_words = list(sorted_neg_stemmed_bag_of_words.keys())
# print (keys_sorted_stemmed_bag_of_words)
# print (keys_sorted_neg_stemmed_bag_of_words)
# print (keys_sorted_pos_stemmed_bag_of_words)

# Preparando array pro knn
set_treino = set_treino_tweets_negativos + set_treino_tweets_positivos
set_teste = set_teste_tweets_negativos + set_teste_tweets_positivos
random.shuffle(set_treino)
random.shuffle(set_teste)
# print (set_treino)
# print(set_teste)
used_words = qtidade_palavras_bag_of_words
bag_of_pos_words_pro_knn = keys_sorted_pos_stemmed_bag_of_words[:used_words]
bag_of_neg_words_pro_knn = keys_sorted_neg_stemmed_bag_of_words[:used_words]
bag_of_words_pro_knn = keys_sorted_stemmed_bag_of_words[:used_words]
bbag_of_words_pro_knn = bag_of_neg_words_pro_knn + bag_of_pos_words_pro_knn
# print(bag_of_words_pro_knn)

set_treino_pro_knn = []
set_treino_pro_weka = []

for linha in set_treino:
    linha_pro_knn = []
    linha_pro_weka = []

    tweet = linha[1]
    tokens = nltk.word_tokenize(tweet)
    tweet_words=[]
    for token in tokens:
        if token.lower() not in stopwords:
            tweet_words.append(stemmer.stem(token.lower()))
    # print (words)

    for word in bag_of_words_pro_knn:
        if word in tweet_words:
            linha_pro_knn.append(1)
            linha_pro_weka.append(1)
        else:
            linha_pro_knn.append(0)
            linha_pro_weka.append(0)

    avaliacao = linha[0]
    if avaliacao == "1":
        linha_pro_knn.append(1)
        linha_pro_weka.append("positivo")
    else:
        linha_pro_knn.append(0)
        linha_pro_weka.append("negativo")

    set_treino_pro_knn.append(linha_pro_knn)
    set_treino_pro_weka.append(linha_pro_weka)
    # print (linha_pro_knn)

# print (set_treino_pro_knn)
# print (set_treino_pro_weka)

contador_tweets_positivos = 0
contador_tweets_negativos = 0
for linha in set_treino_pro_knn:
    if linha[len(linha)-1]==0:
        contador_tweets_negativos +=1
    else:
        contador_tweets_positivos +=1
# print (contador_tweets_negativos)
# print (contador_tweets_positivos)



set_teste_pro_knn = []
set_teste_pro_weka = []

for linha in set_teste:
    linha_pro_knn = []
    linha_pro_weka = []

    tweet = linha[1]
    tokens = nltk.word_tokenize(tweet)
    tweet_words=[]
    for token in tokens:
        if token.lower() not in stopwords:
            tweet_words.append(stemmer.stem(token.lower()))
    # print (words)

    for word in bag_of_words_pro_knn:
        if word in tweet_words:
            linha_pro_knn.append(1)
            linha_pro_weka.append(1)
        else:
            linha_pro_knn.append(0)
            linha_pro_weka.append(0)

    avaliacao = linha[0]
    if avaliacao == "1":
        linha_pro_knn.append(1)
        linha_pro_weka.append("positivo")
    else:
        linha_pro_knn.append(0)
        linha_pro_weka.append("negativo")

    set_teste_pro_knn.append(linha_pro_knn)
    set_teste_pro_weka.append(linha_pro_weka)
    # print (linha_pro_knn)


# print (set_teste_pro_knn)
# print (set_teste_pro_weka)

contador_tweets_positivos = 0
contador_tweets_negativos = 0
for linha in set_teste_pro_knn:
    if linha[len(linha)-1]==0:
        contador_tweets_negativos +=1
    if linha[len(linha)-1]==1:
        contador_tweets_positivos +=1
# print (contador_tweets_negativos) # 82
# print (contador_tweets_positivos) # 40



def euclidiana(atual, amostra):
    soma = 0;
    for i in range(len(atual)):
        soma = soma + (atual[i] - amostra[i]) * (atual[i] - amostra[i])
    return math.sqrt(soma)


def ordena(distancia):
    aux = []
    for i in len(distancia):
        for j in len(distancia):
            if (distancia[j][0] > distancia[j + 1][0]):
                troca(distancia[j], distancia[j + 1])


def moda(dist, k):
    rotulo = 0
    cont = 0
    avaliacao = -1
    quant = 0
    for c in range(2):
        cont = 0
        for i in range(k):
            if (dist[i][1]==c):
                cont += 1
        if cont > quant:
            quant = cont
            avaliacao = c
    return avaliacao



def troca(linha1, linha2):
    for i in range(len(linha1)):
        aux = linha1[i]
        linha1[i] = linha2[i]
        linha2[i] = aux

# define o número de vizinhos
k = k_vizinhos

acertos=0
positivo_classificado_como_positivo = 0
positivo_classificado_como_negativo = 0
negativo_classificado_como_positivo = 0
negativo_classificado_como_negativo = 0
def executaKnn():
    acertos = 0
    positivo_classificado_como_positivo = 0
    positivo_classificado_como_negativo = 0
    negativo_classificado_como_positivo = 0
    negativo_classificado_como_negativo = 0
    conta_tweets_pos = 0
    conta_tweets_neg = 0
    for i in range((len(set_teste_pro_knn))):
        distancia = [[0 for x in range(2)] for y in range(482)]
        for j in range((len(set_treino_pro_knn))-1):
            distancia[j][0] = euclidiana(set_teste_pro_knn[i], set_treino_pro_knn[j])
            # distancia[j][1] = set_treino_pro_knn[j][used_words]
            distancia[j][1] = set_treino_pro_knn[j][len(set_treino_pro_knn[i]) - 1]

        distancia.sort()
        avaliacao_predita = moda(distancia, k)
        avaliacao_real = set_teste_pro_knn[i][len(set_teste_pro_knn[i])-1]
        if (avaliacao_real==0):
            conta_tweets_neg = conta_tweets_neg + 1
        if (avaliacao_real==1):
            conta_tweets_pos = conta_tweets_pos + 1
        # print("Avaliação Predita:", avaliacao_predita)
        # print("Avaliação Real:", avaliacao_real)
        if (avaliacao_real==avaliacao_predita):
            acertos+=1
        if (avaliacao_real==0 and avaliacao_predita==0):
            negativo_classificado_como_negativo = negativo_classificado_como_negativo+1
        if (avaliacao_real==0 and avaliacao_predita==1):
            negativo_classificado_como_positivo = negativo_classificado_como_positivo+1
        if (avaliacao_real==1 and avaliacao_predita==0):
            positivo_classificado_como_negativo = positivo_classificado_como_negativo+1
        if (avaliacao_real==1 and avaliacao_predita==1):
            positivo_classificado_como_positivo = positivo_classificado_como_positivo+1
    print("Positivos avaliados como positivos =", positivo_classificado_como_positivo)
    print("Positivos avaliados como negativos =", positivo_classificado_como_negativo)
    print("Negativos avaliados como positivos =", negativo_classificado_como_positivo)
    print("Negativos avaliados como negativos =", negativo_classificado_como_negativo)
    print("Quantidade total de testes", len(set_teste_pro_knn))
    print("Quantidade total de acertos", acertos)
    #print("Tweets pos", conta_tweets_pos)
    #print("Tweets neg", conta_tweets_neg)
    print ("Em relação aos tweets positivos")
    tp = positivo_classificado_como_positivo
    tn = negativo_classificado_como_negativo
    fp = negativo_classificado_como_positivo
    fn = positivo_classificado_como_negativo
    print("TP:", tp, " TN:", tn, " FP:", fp, " FN:", fn)
    print("Precision:", (tp / (tp+fp)))
    print("Recall:", (tp / (tp+fn)))
    print("F-Measure:", 2*(tp / (tp+fp))*(tp / (tp+fn)) / (tp / ((tp+fp)) + (tp / (tp+fn))))

    return acertos


acertos = executaKnn()
# print ("Acertos", acertos)
# print ("Testes", len(set_teste_pro_knn))

with open('tweetsarrf.txt', 'w', encoding='utf-8') as f:
    f.write("% 1. Title: Tweets evaluation" + "\n")
    f.write("%" + "\n")
    f.write("% 2. Source:" + "\n")
    f.write("%      (a) Creator: Roberto Rezende" + "\n")
    f.write("%      (b) Donor: https://github.com/silviawmoraes/PLN/tree/master/corpora/computer-br" + "\n")
    f.write("%      (c) Date: Juny, 2021" + "\n")
    f.write("%" + "\n")
    f.write("@RELATION tweets" + "\n")
    f.write("\n")
    for word in bag_of_words_pro_knn:
            if len(word) > 11:
                linha = "@ATTRIBUTE " + word + "\t" + "NUMERIC" + "\n"
            elif len(word) > 4:
                linha = "@ATTRIBUTE " + word + "\t" + "\t" + "NUMERIC" + "\n"
            else:
                linha = "@ATTRIBUTE " + word + "\t" + "\t" + "\t" + "NUMERIC" + "\n"
            f.write(linha)

    f.write("@ATTRIBUTE class" + "\t" + "\t" + "{negativo, positivo}" + "\n")
    f.write("@DATA" + "\n")
    for tweet in set_treino_pro_weka:
        linha = ', '.join(map(str, tweet))
        f.write(linha + "\n")
    for tweet in set_teste_pro_weka:
        linha = ', '.join(map(str, tweet))
        f.write(linha + "\n")


with open('tweets_teste.arff', 'w', encoding='utf-8') as f:
    f.write("% 1. Title: Tweets evaluation" + "\n")
    f.write("%" + "\n")
    f.write("% 2. Source:" + "\n")
    f.write("%      (a) Creator: Roberto Rezende" + "\n")
    f.write("%      (b) Donor: https://github.com/silviawmoraes/PLN/tree/master/corpora/computer-br" + "\n")
    f.write("%      (c) Date: Juny, 2021" + "\n")
    f.write("%" + "\n")
    f.write("@RELATION tweets" + "\n")
    f.write("\n")
    for word in bag_of_words_pro_knn:
            if len(word) > 11:
                linha = "@ATTRIBUTE " + word + "\t" + "NUMERIC" + "\n"
            elif len(word) > 4:
                linha = "@ATTRIBUTE " + word + "\t" + "\t" + "NUMERIC" + "\n"
            else:
                linha = "@ATTRIBUTE " + word + "\t" + "\t" + "\t" + "NUMERIC" + "\n"
            f.write(linha)

    f.write("@ATTRIBUTE class" + "\t" + "\t" + "{negativo, positivo}" + "\n")
    f.write("@DATA" + "\n")
    for tweet in set_teste_pro_weka:
        linha = ', '.join(map(str, tweet))
        f.write(linha + "\n")
