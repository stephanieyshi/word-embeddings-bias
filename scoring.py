import json
import numpy as np
from sklearn.decomposition import PCA

def weat():
    pass


def get_embedding_dict(filename):
	with open(filename, 'r') as f:
		lines = f.read().strip().split('\n')

	embedding_dict = {}

	for line in lines:
		split_line = line.split(' ')
		word = split_line[0]
		vector = np.array([float(x) for x in split_line[1:]])
		embedding_dict[word] = vector

	return embedding_dict


def read_json(filename):
	with open(filename, 'r') as f:
		return json.load(f)


def get_pca(pairs, embedding_dict):
	X = []

	for pair in pairs:
		center = 0.5 * (embedding_dict[pair[0]] + embedding_dict[pair[1]])
		X.append(embedding_dict[pair[0]] - center)
		X.append(embedding_dict[pair[1]] - center)

	pca = PCA(n_components=10)
	components = pca.fit(X)

	return pca


def get_gender_direction(embedding_dict, pairs_file):
	pairs = read_json(pairs_file)
	pca = get_pca(pairs, embedding_dict)
	
	#print(pca.explained_variance_ratio_)
	return pca.components_[0]


def get_neutral_embeddings(embedding_dict, gender_specific_words):
	neutral_words = []
	neutral_embeddings = []
	for word in embedding_dict:
		if word not in gender_specific_words:
			neutral_words.append(word)
			vectors.append(embedding_dict[word])

	return neutral_words,neutral_embeddings


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


		



def main():
	embedding_dict = get_embedding_dict('embeddings/w2v_gnews_small.txt')
	g = get_gender_direction(embedding_dict, 'data/definitional_pairs.json')
	gender_specific_words = read_json('data/gender_specific_full.json')
	gender_neutral_words = [word for word in embedding_dict if word not in gender_specific_words]
	professions = [data[0] for data in read_json('data/professions.json')]

	direct_bias = get_direct_bias(embedding_dict, g, professions)
	print("Direct Bias: " + str(direct_bias))

	#neutral_pairs = read_json('data/equalize_pairs.json')
	#indirect_bias = get_indirect_bias(embedding_dict, g, neutral_pairs)
	#print("Indirect Bias: " + str(indirect_bias))
	






if __name__ == '__main__':
    main()
