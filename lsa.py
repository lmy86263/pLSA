from scipy import linalg
import numpy as np
import glob
from preprocess import read_data, pre_process, word_doc_matrix


def svd(X):
    u, sigma, v = linalg.svd(X, full_matrices=True)
    return u, sigma, v


def reduce_dimension(X, number_of_topics):
    u, sigma, v = svd(X)
    reduced_u = u[0:, 0:number_of_topics]
    reduced_v = v[0:number_of_topics, 0:]
    reduced_sigma = np.diag(sigma[0:number_of_topics])
    print(reduced_u.shape, reduced_v.shape, reduced_sigma.shape)
    return reduced_u, reduced_sigma, reduced_v


def lsa(X, number_of_topics):
    reduced_u, reduced_sigma, reduced_v = reduce_dimension(X, number_of_topics)
    word_topic_matrix = np.dot(reduced_u, reduced_sigma)
    topic_doc_matrix = np.dot(reduced_sigma, reduced_v)

    app_X = np.dot(np.dot(reduced_u, reduced_sigma), reduced_v)
    return word_topic_matrix, topic_doc_matrix, app_X


if __name__ == '__main__':
    files = glob.glob('./text/*.txt')
    documents = []
    for f in files:
        documents.append(read_data(f))

    documents, words = pre_process(documents)
    X = word_doc_matrix(words, documents)

    word_topic_matrix, topic_doc_matrix, app_X = lsa(X, 5)
    print(word_topic_matrix)
    print(topic_doc_matrix)
    print(app_X)








