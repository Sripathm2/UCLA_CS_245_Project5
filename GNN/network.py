import numpy as np
from scipy import sparse
import tensorflow as tf
import spektral
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.datasets.mnist import MNIST

data = MNIST()


class Net(tf.keras.Model):
    def __init__(self, window=6, dropout=.5, **kwargs):
        """
        Window: int. Window of days
        #LSTM hidden states: 64
        Training: 500 epocs, batchsize 8, Adam optimizer, LR 10-3
        """
        super().__init__(**kwargs)
        self._nets = {}
        for i in range(1,window+1):
            self.build_MPNN_unit(dropout, i)
        self.LSTM1 = tf.keras.layers.LSTM(, return_sequences=True)
        self.LSTM2 = tf.keras.layers.LSTM(, return_sequences=False,
                                          return_state=True)

    def build_MPNN_unit(self, dropout, net_id=1):
        L1 = []
        L1.append(
            spektral.layers.MessagePassing(aggregate='sum',
                                           activation='relu')
            )
        L1.append(
            tf.keras.layers.BatchNormalization()
            )
        L1.append(
            tf.keras.layers.Dropout(dropout)
            )
        L2 = []
        L2.append(
            spektral.layers.MessagePassing(aggregate='sum',
                                           activation='relu')
            )
        L2.append(
            tf.keras.layers.BatchNormalization()
            )
        L2.append(
            tf.keras.layers.Dropout(dropout)
            )
        self._nets[net_id] = (L1, L2)


    def run_MPNN_unit(self, Adj, X, net_id=1):
        """
        Function calls layer given by id
        """
        L1, L2 = self._nets[net_id]
        y = None
        for i in range(0,len(L1)):
            if i == 0: # MessagePassing layer
                y = L1[i].propagate(X, Adj)
                continue
            print(i,L1[i])#, y)
            y = L1[i](y)
        H1 = y
        for i in range(0, len(L2)):
            if i == 0: # MessagePassing Layer
                y = L2[i].propagate(y, Adj)
                continue
            y = L2[i](y)
        H2 = y
        return tf.concat((H1,H2), axis=1)
    
    def call(self, Adj, X):
        print(self._nets)
        H1 = self.run_MPNN_unit(Adj[0], X[0], net_id=1)
        H2 = self.run_MPNN_unit(Adj[1], X[1], net_id=2)
        H3 = self.run_MPNN_unit(Adj[2], X[2], net_id=3)
        H4 = self.run_MPNN_unit(Adj[3], X[3], net_id=4)
        H5 = self.run_MPNN_unit(Adj[4], X[4], net_id=5)
        H6 = self.run_MPNN_unit(Adj[5], X[5], net_id=6)
        LSTM_input = [H1, H2, H3, H4, H5, H6]
        for i in range(0,len(LSTM_input)):
            LSTM_input[i] = tf.expand_dims(LSTM_input[i], 0)
        x = self.LSTM1(LSTM_input)
        x = self.LSTM2(x)
        x = X+x
        #Lin?
        x = tf.keras.activations.relu(x)
        return x

Adj = np.zeros((52,52))
Adj[5][5] = 1
Adj[3][5] = 2
Adj = sp_matrix_to_sp_tensor(Adj)
ad = [Adj for i in range(0,6)]
"""
adj_list = []
for i in range(0, Adj.shape[0]):
    adj_list.append(sp_matrix_to_sp_tensor(Adj[i][:][:]))
Adj = sparse.hstack(adj_list)
"""
print(Adj.shape)
X = [np.random.rand(52, 5) for i in range(0,6)]
y = np.random.rand(52, 1)
mod = Net(window=6)
mod(ad, X)
