# Author : Sarath Chandar


import os
import sys
import time
from numpy import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import sparse
import progressbar

"""
This function "load" takes 3 inputs : name of the file to be loaded, shared variable to which the file is to be loaded, number of columns in the matrix
Number of rows in matrix is fixed to be 1000. Number of columns in determined by the number of nodes in the visible layers.
The matrix file should be in scipy sparse matrix format.
"""
def load(filename):
    print "loading ... ", filename
    mm = sparse.load_npz(filename).astype(theano.config.floatX)
    return mm

class Autoencoder(object):

    def __init__(self, numpy_rng, theano_rng=None, input=None,n_visible=400, n_hidden=200,W=None, bhid=None, bvis=None,fts1=100,fts2=100,lamda = 4):

        
        # Set the number of visible units and hidden units in the network
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            print "randomly initializing W"
            initial_W = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        else:
            print "loading W matrix"
            initial_W = numpy.load(W+".npy")
            W = theano.shared(value=initial_W, name='W', borrow=True)
      

        if not bvis:
            print "randomly initializing visible bias"
            bvis = theano.shared(value=numpy.zeros(n_visible,dtype=theano.config.floatX),borrow=True)

        else:
            print "loading visible bias"
            initial_bvis = numpy.load(bvis+".npy")
            bvis = theano.shared(value=initial_bvis, name='bvis', borrow=True)
 
        if not bhid:
            print "randomly initializing hidden bias"
            bhid = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='bhid',borrow=True)

        else:
            print "loading hidden bias"
            initial_bhid = numpy.load(bhid+".npy")
            bhid = theano.shared(value=initial_bhid, name='bhid', borrow=True)


        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.fts1 = fts1
        self.fts2 = fts2
        self.lamda = lamda
        

        self.theano_rng = theano_rng
        
        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
            

        self.params = [self.W,  self.b, self.b_prime]

    

    """
    input_type = 0 means parallel data point.
    input_type = 1 means data point with first view only.
    input_type = 2 means data point with second view only.
    """

    def get_cost_updates(self, input_type, learning_rate):

        if(input_type==0):
            print "setting up parallel objective"

            y1 = (T.dot(self.x[:,self.fts1:self.n_visible], self.W[self.fts1:self.n_visible,:]) + self.b)
            z1 = (T.dot(T.nnet.sigmoid(y1), self.W_prime) + self.b_prime)

            y2 = (T.dot(self.x[:,0:self.fts1], self.W[0:self.fts1,:]) + self.b)
            z2 = (T.dot(T.nnet.sigmoid(y2), self.W_prime) + self.b_prime)

            
            y3 = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)
            z3 = (T.dot(y3, self.W_prime) + self.b_prime)


            cor = list()
            batch_size = y1.shape[0]

            for i in range(0,self.n_hidden):
                x1 = y1[:,i] - (T.ones((batch_size,))*(T.sum(y1[:,i])/batch_size))
            	x2 = y2[:,i] - (T.ones((batch_size,))*(T.sum(y2[:,i])/batch_size))
            	nr = T.sum(x1 * x2) / (T.sqrt(T.sum(x1 * x1))*T.sqrt(T.sum(x2 * x2)))
            	cor.append(-nr)
            

            # L1, L2, L3 are cross entrophy loss. If you have data which is not binary, use a different loss function like squarred error loss.    
            L1 = - T.sum(self.x * T.log(T.nnet.sigmoid(z1)) + (1 - self.x) * T.log(1 - T.nnet.sigmoid(z1)), axis=1)
            L2 = - T.sum(self.x * T.log(T.nnet.sigmoid(z2)) + (1 - self.x) * T.log(1 - T.nnet.sigmoid(z2)), axis=1)
            L3 = - T.sum(self.x * T.log(T.nnet.sigmoid(z3)) + (1 - self.x) * T.log(1 - T.nnet.sigmoid(z3)), axis=1)
            L4 =  T.sum(cor)
            L = L1 + L2 + L3 + (self.lamda * L4) + 100

            cost = T.mean(L)

            gparams = T.grad(cost, self.params)
            updates = []
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - learning_rate * gparam))

            return (cost, updates)

        elif(input_type==1):

            print "setting up view-1 objective"
            
            y2 = T.nnet.sigmoid(T.dot(self.x[:,0:self.fts1], self.W[0:self.fts1,:]) + self.b)
            z2 = T.nnet.sigmoid(T.dot(y2, self.W_prime) + self.b_prime)
            
            # L is a squared error loss
            L = T.sum(T.sqr(self.x[:,0:self.fts1]-z2[:,0:self.fts1])/2,axis = 1)
            
            cost = T.mean(L)

            gparams = T.grad(cost, self.params)
            updates = []
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - learning_rate * gparam))

            return (cost, updates)        

        elif(input_type==2):

            print "setting up view-2 objective"

            y2 = T.nnet.sigmoid(T.dot(self.x[:,self.fts1:self.n_visible], self.W[self.fts1:self.n_visible,:]) + self.b)
            z2 = T.nnet.sigmoid(T.dot(y2, self.W_prime) + self.b_prime)

            # L is a squared error loss
            L = T.sum(T.sqr(self.x[:,self.fts1:self.n_visible]-z2[:,self.fts1:self.n_visible])/2,axis = 1)

            cost = T.mean(L)

            gparams = T.grad(cost, self.params)
            updates = []
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - learning_rate * gparam))

            return (cost, updates)        



    # This method saves W, bvis and bhid matrices. `n` is the string attached to the file name. 
    def save_matrices(self,n):

        numpy.save("results/b"+n, self.b.get_value(borrow=True))
        numpy.save("results/bp"+n, self.b_prime.get_value(borrow=True))
        numpy.save("results/w"+n, self.W.get_value(borrow=True))

        
def train_test_split(data):
    idx = int(data.shape[0]*0.8)
    return data[:idx,:], data[idx:,:]

def corr_net(data, learning_rate=0.1, training_epochs=50,
            batch_size=20, nvis=400,nhid=40,fts1=100,fts2=100, lamda = 4):

    index = T.lscalar()   
    x = T.matrix('x') 

    train_data, val_data = train_test_split(data)
    train_samples = train_data.shape[0]
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = Autoencoder(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=nvis, n_hidden=nhid,fts1=fts1, fts2=fts2, lamda = lamda)

    print "Model built"

    
    train_x = T.matrix(name='train_x', dtype=theano.config.floatX)
    
    cost3, updates3 = da.get_cost_updates(input_type=0,learning_rate=learning_rate)
    train_daxy = theano.function([train_x], cost3,updates=updates3, givens={x:train_x})
    val_daxy = theano.function([train_x], cost3, givens={x:train_x})

    print "Cost and update functions generated"
    

    flag = 1
    num_train_batches = 1+train_data.shape[0]//batch_size
    num_val_batches = 1+val_data.shape[0]//batch_size

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        
        train_loss = 0.0
        bar = progressbar.ProgressBar()
        for batch_index in bar(range(0,num_train_batches)):
            lo_idx = batch_size*batch_index
            hi_idx = min(batch_size*(batch_index+1), train_data.shape[0])
            train_loss += train_daxy(train_data[lo_idx:hi_idx].todense())

        print("Train loss = %.2f" % (train_loss / num_train_batches))

        val_loss = 0.0
        for batch_index in range(0,num_val_batches):
            lo_idx = batch_size*batch_index
            hi_idx = min(batch_size*(batch_index+1), val_data.shape[0])
            val_loss += val_daxy(val_data[lo_idx:hi_idx].todense())
        print("val loss = %.2f" % (val_loss / num_val_batches))

        if((epoch+1)%2==0):
            da.save_matrices(str(epoch))


    da.save_matrices("final")
