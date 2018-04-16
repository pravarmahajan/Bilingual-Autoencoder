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


"""
This function "load" takes 3 inputs : name of the file to be loaded, shared variable to which the file is to be loaded, number of columns in the matrix
Number of rows in matrix is fixed to be 1000. Number of columns in determined by the number of nodes in the visible layers.
The matrix file should be in scipy sparse matrix format.
"""
def load(file,train_set_x,nvis):
    print "loading ... ", file
    x = numpy.load(file+"d.npy")
    y = numpy.load(file+"i.npy")
    z = numpy.load(file+"p.npy")
    mm = sparse.csr_matrix((x,y,z),shape=(1000,nvis))
    mm = sparse.load_npz(file)
    mm = mm.todense()
    train_set = numpy.array(mm,dtype="float32")
    train_set_x.set_value(train_set,borrow=True)



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

            for i in range(0,self.n_hidden):
                x1 = y1[:,i] - (ones(20)*(T.sum(y1[:,i])/20))
            	x2 = y2[:,i] - (ones(20)*(T.sum(y2[:,i])/20))
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

        
def corr_net(learning_rate=0.1, training_epochs=50,
            batch_size=20, nvis=400,nhid=40,fts1=100,fts2=100, lamda = 4):

    import ipdb; ipdb.set_trace()
    index = T.lscalar()   
    x = T.matrix('x') 

    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = Autoencoder(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=nvis, n_hidden=nhid,fts1=fts1, fts2=fts2, lamda = lamda)

    ct = 0
    
    
    start_time = time.clock()

    train_set_x = theano.shared(numpy.asarray(numpy.zeros((1000,nvis)), dtype=theano.config.floatX), borrow=True)
    
    cost1, updates1 = da.get_cost_updates(input_type=1,learning_rate=learning_rate)
    train_dax = theano.function([index], cost1,updates=updates1,givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})
        
    cost2, updates2 = da.get_cost_updates(input_type=2,learning_rate=learning_rate)
    train_day = theano.function([index], cost2,updates=updates2,givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})
    
    cost3, updates3 = da.get_cost_updates(input_type=0,learning_rate=learning_rate)
    train_daxy = theano.function([index], cost3,updates=updates3,givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})
    
    

    typeflag=0
    diff = 0
    flag = 1

    detfile = file("details.txt","w")
    detfile.close()

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        
        c = []
        
        ipfile = open("ip.txt","r")

        for line in ipfile:
            next = line.strip().split(",")
            load(next[1],train_set_x,nvis)
            if(next[0]=="x"):
                typeflag = 1            
            elif(next[0]=="y"):
                typeflag = 2
            else:
                typeflag = 0    
        
            for batch_index in range(0,int(next[2])):
                if(typeflag==0):
                    c.append(train_daxy(batch_index))
                elif(typeflag==1):
                    c.append(train_dax(batch_index))
                elif(typeflag==2):
                    c.append(train_day(batch_index))


        if(flag==1):
            flag = 0
            diff = numpy.mean(c)
            di = diff
        else:
            di = numpy.mean(c) - diff
            diff = numpy.mean(c)
            print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        detfile = file("details.txt","a")
        detfile.write(str(diff)+"\n")
        detfile.close()
        # save the parameters for every 2 epochs
        if((epoch+1)%2==0):
            da.save_matrices(str(epoch))

    end_time = time.clock()

    training_time = (end_time - start_time)

    print ' code ran for %.2fm' % (training_time / 60.)
    da.save_matrices("final")

                

