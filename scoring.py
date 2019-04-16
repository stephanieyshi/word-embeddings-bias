import json
import numpy as np
from sklearn.decomposition import PCA

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
		w = embedding_dict[pair[0]]
		v = embedding_dict[pair[1]]
		w_g = np.dot(w, g) * w
		v_g = np.dot(v, g) * v

		l1 = np.sqrt(np.dot(w - w_g, w - w_g))
		l2 = np.sqrt(np.dot(v - v_g, v - v_g))

		beta = (1 / np.dot(w, v)) * (np.dot(w, v) - (np.dot(w - w_g, v - v_g) / (l1 * l2)))
		bias[(pair[0], pair[1])] = beta

	return bias
