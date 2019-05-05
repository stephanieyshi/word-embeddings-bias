import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.stats import pearsonr
from scipy import stats
import random
import csv

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


def get_gender_direction(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    vector_string = lines[0]
    vector = np.array([float(val) for val in vector_string.split(' ')])
    return vector


def get_words(filename):
    words = list()
    with open(filename, 'r') as f:
        lines = f.read().split('\n')

    return lines[:len(lines) - 1]


def read_wordsim(filename):
    pairs_dict = dict()
    with open(filename) as csvfile:
        #skip first row
        next(csvfile)
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            pairs_dict[(row[0], row[1])] = float(row[2])

    return pairs_dict


def compute_cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    l1 = np.sqrt(np.dot(v1, v1))
    l2 = np.sqrt(np.dot(v2, v2))

    return dot / (l1 * l2)


# Computes the Pearson correlation between vectors of biases
# v1: unbiased biases
# v2: de-biased biases
def get_pearson_correlation(v1, v2):
    corr, p_value = pearsonr(v1, v2)
    return corr, p_value


# Returns a vector of biases given an embedding_dict and gender direction g
def get_biases(embedding_dict, g, words, c=1):
    biases = []
    for word in words:
        biases.append(abs(compute_cosine_similarity(embedding_dict[word], g)) ** c)

    return biases


# Deprecated? Can get mean of the actual vector of biases
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


def get_kmeans(points):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    labels = kmeans.labels_
    return labels


def plotPCA(embedding_dict, words):
    X = []

    for word in words:
        X.append(embedding_dict[word])

    pca = PCA(n_components=2)
    points = pca.fit_transform(X)

    labels = get_kmeans(X)

    cdict = {i:['red','*'] if i < 500 else ['blue', 'o'] for i in range(len(X))}

    fig,ax = plt.subplots(figsize=(8,8))
    for i in range(len(X)):
        ax.scatter(points[i, 0], points[i, 1], c=cdict[i][0], marker=cdict[i][1])

    for i in range(10):
        ax.annotate(words[i], (points[i, 0], points[i, 1]))

    #uncomment to see PCA plot with clustering
    #plt.show()

    return labels


def get_clustering_accuracy(labels):
    correct_labels = [1 if i < 500 else 0 for i in range(len(labels))]
    correct = sum(correct_labels == labels)
    accuracy = correct / len(labels)

    return max(accuracy, 1 - accuracy)


def get_similarities(embedding_dict, pairs):
    return {pair:compute_cosine_similarity(embedding_dict[pair[0]], embedding_dict[pair[1]]) for pair in pairs}


def get_analogies(filename):
    analogies = list()
    with open(filename, 'r') as f:
        lines = f.read().split('\n')

    for line in lines:
        analogies.append(line.split(" "))

    return analogies

def solve_analogy(embedding_dict, a, b, x):
    max_similarity = 0
    most_similar_word = ""
    v = embedding_dict[x] - embedding_dict[a] + embedding_dict[b]

    for word in embedding_dict:
        if word != x:
            w = embedding_dict[word]
            similarity = compute_cosine_similarity(v, w)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_word = word

    return most_similar_word


def solve_all_analogies(embedding_dict, tuples):
    pred_labels = list()

    for tup in tuples:
        pred = solve_analogy(embedding_dict, tup[0], tup[1], tup[2])
        pred_labels.append(pred)

    return pred_labels

def get_analogy_performance(true_labels, pred_labels):
    return sum([1 for i in range(len(true_labels)) if true_labels[i] == pred_labels[i]]) / len(true_labels)


def write_biases_to_file(biases, filename):
    with open(filename, 'w') as f:
        for bias in biases:
            f.write(str(bias) + "\n")


if __name__ == '__main__':
    # data files
    embeddings_file = '../embeddings/glove_small.txt'
    gender_direction_file = '../embeddings/glove_gender_direction.txt'
    professions_file = '../data/professions.json'
    biased_female_file = '../data/glove_biased_female_500.txt'
    biased_male_file = '../data/glove_biased_male_500.txt'
    original_biases_file = '../data/glove_professions_biases.txt'
    word_similarity_file = '../data/combined.csv'
    analogies_file = '../data/google_analogies.txt'

    # collect data
    print("Collecting Data...")
    embedding_dict = get_embedding_dict(embeddings_file)
    g = get_gender_direction(gender_direction_file)
    professions = [val[0] for val in read_json(professions_file)]

    # evaluating bias
    print("Evaluating Bias...")
    biases = get_biases(embedding_dict, g, professions)
    direct_bias = np.mean(biases)
    print("Direct Bias: " + str(direct_bias))

    pairs = [['receptionist', 'softball'], ['waitress', 'softball'], ['homemaker', 'softball']]
    indirect_bias = get_indirect_bias(embedding_dict, g, pairs)
    print("Indirect Bias: " + str(indirect_bias))

    most_biased_words = get_words(biased_female_file) + get_words(biased_male_file)
    labels = plotPCA(embedding_dict, most_biased_words)
    clustering_accuracy = get_clustering_accuracy(labels)
    print("Clustering Accuracy: " + str(clustering_accuracy))
    print()

    original_biases = [float(val) for val in get_words(original_biases_file)]
    pearson_correlation, p_value = get_pearson_correlation(original_biases, biases)
    print("Pearson Correlation: " + str(pearson_correlation) + " with p-value: " + str(p_value))

    # evaluating quality
    print("Evaluating Quality...")
    similarity_pairs = read_wordsim(word_similarity_file)
    similarity_pairs = {key:similarity_pairs[key] for key in similarity_pairs 
                            if key[0] in embedding_dict and key[1] in embedding_dict}
    
    similarity_dict = get_similarities(embedding_dict, list(similarity_pairs.keys()))
    spearman = stats.spearmanr([float(val) for val in similarity_pairs.values()], 
                               [float(val) for val in similarity_dict.values()])
    
    print("Spearman Correlation for WS: " + str(spearman.correlation))
    
    analogies = get_analogies(analogies_file)
    analogies = random.sample([tup for tup in analogies if tup[0] in embedding_dict and tup[1] in embedding_dict and
                                                           tup[2] in embedding_dict and tup[3] in embedding_dict], 100)
    
    pred_labels = solve_all_analogies(embedding_dict, analogies)
    analogy_accuracy = get_analogy_performance([tup[3] for tup in analogies], pred_labels)
    print("Google Analogies Accuracy: " + str(analogy_accuracy))

