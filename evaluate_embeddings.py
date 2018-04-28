import numpy as np
import heapq
import operator
import random

def read_embeddings(filename):
    embeddings = dict()
    with open(filename) as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(' ')
        word = parts[0]
        vector = [float(w) for w in parts[1:]]
        embeddings[word] = np.array(vector)
    return embeddings

def similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-10)

def get_best_emb(embeddings, test_emb):
    scores = []
    for (k, v) in embeddings.items():
        sim = similarity(v, test_emb)
        scores.append((k, sim))
    return heapq.nlargest(10, scores, operator.itemgetter(1))

def main():
    eng_emb = read_embeddings("./word2vec/inuk.txt")
    inuk_emb = read_embeddings("./word2vec/eng.txt")
    test_words = random.sample(inuk_emb.keys(), 10)
    for w in test_words:
        print(w)
        matches = get_best_emb(eng_emb, inuk_emb[w])
        for in_w, sim in matches:
            print("\t(%s, %.2f)" %(in_w, sim))

if __name__ == "__main__":
    main()
