from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

from pattern.en import lemma
import numpy as np

from collections import Counter
from gensim import corpora

import glob


def read_data(file):
    with open(file, 'r') as f:
        content = f.read()

    return content


def pre_process(documents):
    # lower case
    documents = [doc.lower() for doc in documents]

    # remove stop words
    stop_words = set(stopwords.words('english'))
    symbols = {'-', '(', ')', ';', "'", '.', '_', '/', ':', ',', ');', '‘', '’', '"', '“', '”', '—'}
    stop_words = stop_words | symbols
    tokenizer = TweetTokenizer()
    # lemmatization
    wnl = WordNetLemmatizer()
    filter_documents = [' '.join([wnl.lemmatize(w) for w in tokenizer.tokenize(doc) if w not in stop_words])
                        for doc in documents]
    # filter_documents = [' '.join([lemma(w) for w in tokenizer.tokenize(doc) if w not in stop_words])
    #                     for doc in documents]

    texts = [[word for word in d.split(' ')] for d in filter_documents]
    d = corpora.Dictionary(texts)
    words = [w for w in d.token2id.keys()]
    return filter_documents, words


def word_doc_matrix(vocabulary, documents):
    d = dict(zip(vocabulary, range(len(vocabulary))))
    X = np.zeros([len(vocabulary), len(documents)])
    for j, doc in enumerate(documents):
        items_count = Counter(doc.split(' '))
        for k, v in items_count.items():
            X[d[k], j] = v
    return X


if __name__ == '__main__':
    files = glob.glob('./text/*.txt')
    documents = []
    for f in files:
        documents.append(read_data(f))

    documents, words = pre_process(documents)
    X = word_doc_matrix(words, documents)

