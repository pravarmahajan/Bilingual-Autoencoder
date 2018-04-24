import string
import scipy.sparse
import numpy as np

def get_bos_matrix(filename):
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
    return bos_matrix

m1 = get_bos_matrix("./data/eng_sent_data")
print(m1.shape[1])
m2 = get_bos_matrix("./data/inuk_sent_data")
print(m2.shape[1])

scipy.sparse.save_npz('en_iu', scipy.sparse.hstack((m1, m2), format='csr'))
