import string
import scipy.sparse
import numpy as np

filename = "./bible/de0.txt"
lines = open(filename, 'r').readlines()

word_to_id = dict()
y = []
z = np.array([None]*(len(lines)+1))
z[0] = 0

for (i, line) in enumerate(lines):
    cleaned = ''.join([ch for ch in line.lower() if ch not in string.punctuation])
    parts = cleaned.strip().split()[1:]

    for word in enumerate(parts):
        if not word in word_to_id:
            word_to_id[word] = len(word_to_id)
        y.append(word_to_id[word])
    z[i+1] = len(y)   

x = np.ones((len(y),))

bos_matrix = scipy.sparse.csr_matrix((x, np.array(y), z))
scipy.sparse.save_npz(filename.split("/")[-1].split('.')[0], bos_matrix)
