import json
import numpy as np
from sklearn.decomposition import PCA

def get_embedding_dict(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')

    embedding_dict = {}

    for line in lines:
        split_line = line.strip().split(' ')
        if (len(split_line) > 0):
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

    pca = PCA()
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
            if 1 - np.linalg.norm(nu)**2 < 0:
                z = 0

            if np.dot(embedding_dict[v] - embedding_dict[w], g) < 0:
                z = -z

            embedding_dict[v] = normalize(z * g + nu)
            embedding_dict[w] = normalize(-z * g + nu)

    return embedding_dict


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


def write_embeddings_to_file(embedding_dict, filename):
    with open(filename, 'w') as f:
        for word in embedding_dict:
            f.write(word + " ")
            vector = ["{0:.7f}".format(val) for val in embedding_dict[word]]
            vector_string = " ".join(vector)
            f.write(vector_string + "\n")

def write_words_to_file(list, filename):
    with open(filename, 'w') as f:
        for word in list:
            f.write(word + "\n")


def write_g_to_file(g, filename):
    with open(filename, 'w') as f:
        vec_string = " ".join(["{0:.7f}".format(val) for val in g])
        f.write(vec_string)

def main():
    #data files
    embeddings_file = 'embeddings/glove_small.txt'
    definitional_pairs_file = 'data/definitional_pairs.json'
    specific_words_file = 'data/gender_specific_full.json' #non-neutral
    equalize_pairs_file = 'data/equalize_pairs.json'

    #collect data
    print("Collecting Data...")
    embedding_dict = get_embedding_dict(embeddings_file)
    g = get_gender_direction(embedding_dict, definitional_pairs_file)
    gender_specific_words = read_json(specific_words_file)
    gender_neutral_words = [word for word in embedding_dict if word not in gender_specific_words and word.islower()]
    equalize_pairs = read_json(equalize_pairs_file)

    #finding most biased words
    print("Finding Biased Words...")
    female_bias_dict = most_biased(embedding_dict, g, gender_neutral_words, True)
    male_bias_dict = most_biased(embedding_dict, g, gender_neutral_words, False)
    sorted_dict = sorted(female_bias_dict, key=female_bias_dict.get, reverse=True)[:10]
    print(sorted_dict)
    print(sorted(male_bias_dict, key=male_bias_dict.get, reverse=True)[:10])

    # debias embeddings
    print("Debiasing...")
    embedding_dict = debias(embedding_dict, g, gender_neutral_words)
    embedding_dict = equalize(embedding_dict, g, equalize_pairs)

    # files to write to
    gender_direction_file = 'embeddings/articles_sample_politics_direction.txt'
    debiased_embeddings_file = 'embeddings/debiased_articles_sample_embedding_dict_politics.txt'
    female_biased_file = 'political-bias/articles_sample_politics_democrat_debiased_500.txt'
    male_biased_file = 'political-bias/articles_sample_politics_republican_debiased_500.txt'

    # write data to files
    #write_g_to_file(g, gender_direction_file)
    #write_embeddings_to_file(embedding_dict, debiased_embeddings_file)
    #write_words_to_file(sorted(female_bias_dict, key=female_bias_dict.get, reverse=True)[:500], female_biased_file)
    #write_words_to_file(sorted(male_bias_dict, key=male_bias_dict.get, reverse=True)[:500], male_biased_file)


if __name__ == '__main__':
    main()
