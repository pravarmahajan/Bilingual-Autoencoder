#  Author : Sarath Chandar
#
#	This is an implementation of the binary bag of words reconstruction training with merged mini-batches (section 2.1) in
#
#			Sarath Chandar, Stanislas Lauly, Hugo Larochelle, Mitesh Khapra, Balaraman Ravindran, Vikar Raykar, Amrita Saha, "An Autoencoder Approach to Learning Bilingual Word Representations"
#																									accepted to Advances in Neural Information Processing Systems 27, 2014


from correlational_net import *
import pickle

lamda = 4       # The lambda parameter explained in the paper
batch_size=256   # No. of examples per batch
nhid = 128	# number of hidden neurons
data_filename = "./en_iu.npz"
data = load(data_filename)
nvis = data.shape[1]
fts1 = len(pickle.load(open('./data/eng_word2id.pkl', 'rb')))	# number of features in first view
#fts1 = 91808 # number of features in first view
fts2 = nvis-fts1 # number of features in second view
assert(fts2==len(pickle.load(open('./data/inuk_word2id.pkl', 'rb'))))
training_epochs = 20  # No. of training epochs

corr_net(data, nvis=nvis,nhid=nhid,fts1=fts1,fts2=fts2, lamda = lamda, training_epochs=training_epochs,batch_size=batch_size)
