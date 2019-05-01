import json
import numpy as np
from sklearn.decomposition import PCA

from scoring.scoring import *

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

    return pca.components_[0]


def get_neutral_embeddings(embedding_dict, gender_specific_words):
    neutral_words = []
    neutral_embeddings = []
    for word in embedding_dict:
        if word not in gender_specific_words:
            neutral_words.append(word)
            vectors.append(embedding_dict[word])

    return neutral_words,neutral_embeddings


def main():
    #data files
    embeddings_file = 'embeddings/w2v_gnews_small.txt'
    definitional_pairs_file = 'data/definitional_pairs.json'
    specific_words_file = 'data/gender_specific_full.json' #non-neutral
    professions_file = 'data/professions.json'
    biased_female_file = 'data/gnews_biased_female_500.txt'
    biased_male_file = 'data/gnews_biased_male_500.txt'
    word_similarity_file = 'data/combined.csv'
    analogies_file = 'data/google_analogies.txt'

    #collect data
    print("Collecting Data...")
    embedding_dict = get_embedding_dict(embeddings_file)
    g = get_gender_direction(embedding_dict, definitional_pairs_file)
    gender_specific_words = read_json(specific_words_file)
    gender_neutral_words = [word for word in embedding_dict if word not in gender_specific_words]
    professions = [data[0] for data in read_json(professions_file)]

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

    print("Evaluating Quality...")
    similarity_pairs = read_wordsim(word_similarity_file)
    similarity_pairs = {key:similarity_pairs[key] for key in similarity_pairs 
                            if key[0] in embedding_dict and key[1] in embedding_dict}
    
    similarity_dict = get_similarities(embedding_dict, list(similarity_pairs.keys()))
    spearman = stats.spearmanr([float(val) for val in similarity_pairs.values()], 
                               [float(val) for val in similarity_dict.values()])
    
    print("Spearman Correlation for Word Similarity: " + str(spearman.correlation))
    
    analogies = get_analogies(analogies_file)
    analogies = random.sample([tup for tup in analogies if tup[0] in embedding_dict and tup[1] in embedding_dict and
                                                           tup[2] in embedding_dict and tup[3] in embedding_dict], 100)
    
    pred_labels = solve_all_analogies(embedding_dict, analogies)
    analogy_accuracy = get_analogy_performance([tup[3] for tup in analogies], pred_labels)
    print("Google Analogies Accuracy: " + str(analogy_accuracy))


if __name__ == '__main__':
    main()
