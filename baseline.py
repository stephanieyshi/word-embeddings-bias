import json
import numpy as np
from sklearn.decomposition import PCA
from scoring.scoring import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_embedding_dict(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    embedding_dict = {}

    for line in lines:
        split_line = line.strip().split(' ')
        if (len(split_line) > 0):
            word = split_line[0]
            # print(split_line)
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
    pca.fit(X)

    return pca


def get_gender_direction(embedding_dict, pairs_file):
    pairs = read_json(pairs_file)
    filtered_pairs = []
    for (female, male) in pairs:
        if female in embedding_dict and male in embedding_dict:
            filtered_pairs.append([female, male])
    pca = get_pca(filtered_pairs, embedding_dict)

    return pca.components_[0]


def get_neutral_embeddings(embedding_dict, gender_specific_words):
    neutral_words = []
    neutral_embeddings = []
    for word in embedding_dict:
        if word not in gender_specific_words:
            neutral_words.append(word)
            vectors.append(embedding_dict[word])

    return neutral_words,neutral_embeddings

def drop(vector, g):
    return vector - g * np.dot(vector, g) / np.dot(g, g)

def normalize(vector):
    return vector / np.linalg.norm(vector)

def debias(embedding_dict, g, gender_neutral_words):
    for word in gender_neutral_words:
        biased_vector = embedding_dict[word]
        debiased_vector = drop(biased_vector, g)
        embedding_dict[word] = normalize(debiased_vector)

    return embedding_dict

def equalize(embedding_dict, g, pairs):
    all_pairs = {x for e1, e2 in pairs for x in [(e1.lower(), e2.lower()),
                                                  (e1.title(), e2.title()),
                                                  (e1.upper(), e2.upper())]}
    for pair in all_pairs:
        v = pair[0]
        w = pair[1]
        if v in embedding_dict and w in embedding_dict:
            mu = (embedding_dict[v] + embedding_dict[w]) / 2
            nu = drop(mu, g)
            z = np.sqrt(1 - np.linalg.norm(nu)**2)

            if np.dot(embedding_dict[v] - embedding_dict[w], g) < 0:
                z = -z

            embedding_dict[v] = normalize(z * g + nu)
            embedding_dict[w] = normalize(-z * g + nu)

    return embedding_dict

#FIX THIS
def generate_analogies(embedding_dict, g, words):
    analogies_dict = dict()
    for i in range(len(words)):
        word1 = words[i]
        for j in range(i + 1, len(words)):
            word2 = words[j]
            diff_vec = embedding_dict[word1] - embedding_dict[word2]
            similarity = compute_cosine_similarity(g, diff_vec)
            analogies_dict[(word1, word2)] = similarity

    return analogies_dict


def most_biased(embedding_dict, g, words, dir=True):
    bias_dict = dict()

    for word in words:
        vector = embedding_dict[word]
        proj = g * np.dot(vector, g) / np.dot(g, g)

        #direction of projection determines gender
        direction = np.linalg.norm(g - proj) - np.linalg.norm(g) > 0

        if direction == dir:
            bias = -1 * np.linalg.norm(proj)
        else:
            bias = np.linalg.norm(proj)

        bias_dict[word] = bias

    return bias_dict


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

    labels = get_kmeans(points)

    cdict = {i:['red','*'] if i < 500 else ['blue', 'o'] for i in range(len(X))}

    fig,ax = plt.subplots(figsize=(8,8))
    for i in range(len(X)):
        ax.scatter(points[i, 0], points[i, 1], c=cdict[i][0], marker=cdict[i][1])

    #plt.show()

    return labels

def get_clustering_accuracy(labels):
    correct_labels = [1 if i < 500 else 0 for i in range(len(labels))]
    correct = sum(correct_labels == labels)
    accuracy = correct / len(labels)

    return max(accuracy, 1 - accuracy)


def solve_analogy(embedding_dict):
    vector = embedding_dict['hospital'] - embedding_dict['doctor'] + embedding_dict['teacher']

    solution_dict = dict()

    for word in embedding_dict:
        solution_dict[word] = compute_cosine_similarity(vector, embedding_dict[word])

    return solution_dict


def write_embeddings_to_file(embedding_dict, filename):
    with open(filename, 'w') as f:
        for word in embedding_dict:
            f.write(word + " ")
            vector = ["{0:.7f}".format(val) for val in embedding_dict[word]]
            vector_string = " ".join(vector)
            f.write(vector_string + "\n")

def write_to_file(list, filename):
    with open(filename, 'w') as f:
        for word in list:
            f.write(word + "\n")


def main():
    #collect data
    # embedding_dict = get_embedding_dict('embeddings/w2v_gnews_small.txt')
    embedding_dict = get_embedding_dict('embeddings/breitbart_embedding_dict.txt')
    g = get_gender_direction(embedding_dict, 'data/definitional_pairs.json')
    gender_specific_words = read_json('data/gender_specific_full.json')
    gender_neutral_words = [word for word in embedding_dict if word not in gender_specific_words and word.islower()]
    equalize_pairs = read_json('data/equalize_pairs.json')

    #HARD DEBIASING
    # embedding_dict = debias(embedding_dict, g, gender_neutral_words)
    # embedding_dict = equalize(embedding_dict, g, equalize_pairs)
    # write_embeddings_to_file(embedding_dict, 'embeddings/debiased_breitbart_embedding_dict.txt')


    #FINDING MOST BIASED WORDS
    female_bias_dict = most_biased(embedding_dict, g, gender_neutral_words, True)
    male_bias_dict = most_biased(embedding_dict, g, gender_neutral_words, False)
    write_to_file(sorted(female_bias_dict, key=female_bias_dict.get, reverse=True)[:500], 'data/breitbart_biased_female_500.txt')
    write_to_file(sorted(male_bias_dict, key=male_bias_dict.get, reverse=True)[:500], 'data/breitbart_biased_male_500.txt')
    # top 1000 biased words
    most_biased_words = sorted(male_bias_dict, key=male_bias_dict.get, reverse=True)[:500] + sorted(female_bias_dict, key=female_bias_dict.get, reverse=True)[:500]




if __name__ == '__main__':
    main()
