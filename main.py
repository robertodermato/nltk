import os
import nltk
from nltk.corpus import machado
from nltk.corpus import floresta
from nltk import ngrams, FreqDist
from nltk.metrics import *

# This data can be used to train taggers (examples below for the Floresta treebank).
# print(nltk.corpus.mac_morpho.words())
# print (floresta.words())


# método pra simplificar as tags do corpus floresta
def simplify_tag(t):
    if "+" in t:
         return t[t.index("+")+1:]
    else:
         return t


# criando tuplas simplificadas de palavra e tag com o corpus floresta
twords = floresta.tagged_words()
twords = [(w.lower(), simplify_tag(t)) for (w,t) in twords]
# print(twords[:10])
# Pretty printing the tagged words:
# print(' '.join(word + '/' + tag for (word, tag) in twords[:10]))


# Here's a function that takes a word and a specified amount of context (measured in characters), and generates a concordance for that word.
def concordance(word, context=30):
     for sent in floresta.sents():
         if word in sent:
             pos = sent.index(word)
             left = ' '.join(sent[:pos])
             right = ' '.join(sent[pos+1:])
             print('%*s %s %-*s' %
                 (context, left[-context:], word, context, right[:context]))

# concordancia = concordance("dar") # doctest: +SKIP
# print(concordancia)


words = floresta.words()
# print(len(words))
fd = nltk.FreqDist(words)
# print(len(fd))
# print(fd.max())

sentencas_floresta = floresta.sents() # doctest: +NORMALIZE_WHITESPACE
tagged_sentecas_floresta = floresta.tagged_sents() # doctest: +NORMALIZE_WHITESPACE
parsed_floresta_sent = floresta.parsed_sents()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
# print (sentencas_floresta)
# print (tagged_sentecas_floresta)
# parsed_floresta_sent[5].draw() # doctest: +SKIP

#tokens
texto = "Oi. Sou o Sr. Roberto. Tudo bem Roberto? Como vão as coisas, hein Roberto?"
tokens = nltk.word_tokenize(texto)
# print(tokens)

#stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')
#print (stopwords[:10])

# Machado de Assis
# print(machado.fileids()[:5])
# text1 = machado.words('romance/marm05.txt')
# print(text1)

# cria um dicionário com palavras e suas frequencias de distribuição ignorando as stopwords
frequencia_de_dist = nltk.FreqDist(w.lower() for w in tokens if w not in stopwords)
#for word in list(frequencia_de_dist.keys()):
#    print(word, frequencia_de_dist[word])

print("Palavra mais comum é ", frequencia_de_dist.max())


# Let's begin by getting the tagged sentence data, and simplifying the tags as described earlier.
tsents = floresta.tagged_sents()
tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent] for sent in tsents if sent]
train = tsents[100:]
test = tsents[:100]
# We already know that n is the most common tag, so we can set up a default tagger that tags every word as a noun, and see how well it does:
tagger0 = nltk.DefaultTagger('n')
#print(accuracy(tagger0, test))
#nltk.tag.accuracy(tagger0, test)

# 0.17697228144989338
# Evidently, about one in every six words is a noun. Let's improve on this by training a unigram tagger:
tagger1 = nltk.UnigramTagger(train, backoff=tagger0)
#print(accuracy(tagger1, test))
#print(nltk.tag.accuracy(tagger1, test))
# 0.87029140014214645
# Next a bigram tagger:
tagger2 = nltk.BigramTagger(train, backoff=tagger1)
#print(accuracy(tagger2, test))
#print(nltk.tag.accuracy(tagger2, test))
# 0.89019189765458417

# Punkt is a language-neutral sentence segmentation tool.
sent_tokenizer=nltk.data.load('tokenizers/punkt/portuguese.pickle')
sentences = sent_tokenizer.tokenize(texto)
for sent in sentences:
     print("<<", sent, ">>")

# NLTK's data collection includes a trained model for Portuguese sentence segmentation,
# which can be loaded as follows. It is faster to load a trained model than to retrain it.
stok = nltk.data.load('tokenizers/punkt/portuguese.pickle')
print("ok")