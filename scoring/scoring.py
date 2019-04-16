import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

def compute_cosine_similarity(v1, v2):
  dot = np.dot(v1, v2)
  l1 = np.sqrt(np.dot(v1, v1))
  l2 = np.sqrt(np.dot(v2, v2))

  return dot / (l1 * l2)


def get_direct_bias(embedding_dict, g, words, c=1):
	bias = 0
	N = len(words)

	for word in words:
		bias += abs(compute_cosine_similarity(embedding_dict[word], g)) ** c

	return (1 / N) * bias


def get_indirect_bias(embedding_dict, g, pairs):
    bias = dict()

    for pair in pairs:
        w = embedding_dict[pair[0]] / np.linalg.norm(embedding_dict[pair[0]])
        v = embedding_dict[pair[1]] / np.linalg.norm(embedding_dict[pair[1]])
        w_g = np.dot(w, g) * g
        v_g = np.dot(v, g) * g

        beta = (1 / np.dot(w, v)) * (np.dot(w, v) - compute_cosine_similarity(w - w_g, v - v_g))
        bias[(pair[0], pair[1])] = beta

    return bias
