import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# This is a class for a one hidden layer FFNN

class FFNN(object):
    # Array
    def __init__(self, nmrOfNodes, lr = 0.01, seed = 0, wr = 0.1):
        
        tf.set_random_seed(seed)
        # Initialising params
        self.nInputNodes = nmrOfNodes[0]
        self.nHiddenNodes = nmrOfNodes[1]
        self.nOutputNodes = nmrOfNodes[2]

        self.lr = lr#tf.Variable(lr, name = 'lr')
        self.seed = seed
        self.wr = wr 
        
        self.state = tf.placeholder(tf.float32, [None, self.nInputNodes])
        
        # Initialise weights and such
        self.wIH = tf.Variable(tf.random_uniform(shape=[self.nInputNodes, self.nHiddenNodes], minval=-self.wr, maxval=self.wr), name='wIH')
        self.bIH = tf.Variable(tf.random_uniform(shape=[self.nHiddenNodes], minval=-self.wr, maxval=self.wr), name='bIH')            
        # Always using bias
        self.hNodes = tf.nn.tanh(tf.matmul(self.state, self.wIH) + self.bIH)

        self.wHO = tf.Variable(tf.random_uniform(shape=[self.nHiddenNodes, self.nOutputNodes], minval=-self.wr, maxval=self.wr), name='wHO')
        self.bHO = tf.Variable(tf.random_uniform(shape=[self.nOutputNodes], minval=-self.wr, maxval=self.wr), name='bHO')

        # Testing to use activation function on output layer (not now)
        self.Q = tf.matmul(self.hNodes, self.wHO) + self.bHO
        
        self.vars = [self.wIH, self.bIH, self.wHO, self.bHO]

        # targets
        self.Qtar = tf.placeholder(tf.float32, [None, self.nOutputNodes])
        # loss
        self.dif = self.Qtar - self.Q
        self.dif = tf.square(self.dif)
        self.mse = tf.reduce_mean(self.dif)
#        self.optimiser = tf.train.GradientDescentOptimizer(self.lr) works like mjeeeh

#       A new test is made with the AdamOptimizer, which automatically lowers lr
        self.optimiser = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train = self.optimiser.minimize(self.mse, var_list=self.vars)
        
        self.error_list = []
    
    def initSession(self):
        model = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(model)

    # 'Run' network and get Q
    def get_Q(self, obs):
        return self.session.run(self.Q, feed_dict={self.state: obs})
    
    def set_lr(self, learning_rate):
        self.lr = learning_rate    
    
    # Make gradient descent
    def gd(self, x_batch, Q_batch):
        _, error = self.session.run([self.train, self.dif], feed_dict={self.state: x_batch, self.Qtar: Q_batch})
        self.error_list.append(error)
        
    def get_params(self):
        params = {}
        for p in self.vars:
            val = self.session.run(p)
            params[p.name] = val
        return params

    def print_params(self):
        params = self.get_params()
        for name,val in params.iteritems():
            print '{}: \n{}\n'.format(name, val)

    def plot_error(self):
        plt.plot([np.mean(self.error_list[i-50:i]) for i in range(len(self.error_list))])
        



