import numpy as np
from scipy import sparse
import tensorflow as tf
import spektral
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.datasets.mnist import MNIST

data = MNIST()


class Net(tf.keras.Model):
    def __init__(self,
                 window=6,
                 lstm_output=52,
                 dropout=.5,
                 batch_size=24,
                 learning_rate=1e-3, **kwargs):
        """
        Initializes a network
        args:
          Window: int. Window of days.
          lstm_output: int. Number of units to the LSTM layer. Output of lstm
          dropout: float. Probabiltiy of dropout.
          batch_size: int. Size of the batch for training
        Training: 500 epocs, batchsize 8, Adam optimizer, LR 10-3
        returns:
          None
        """
        super().__init__(**kwargs)
        self._nets = {}
        self._window = window
        self._batch_size = batch_size
        for i in range(1,window+1):
            self.build_MPNN_unit(dropout, i)
        self.LSTM1 = tf.keras.layers.LSTM(lstm_output, return_sequences=True)
        self.LSTM2 = tf.keras.layers.LSTM(lstm_output, return_state=True)
        self.Lin = tf.keras.layers.Dense(52, input_shape=(6, 52), activation="relu")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

    def build_MPNN_unit(self, dropout, net_id=1):
        """
        Function builds an MPNN unit
        args:
          dropout: float. Probabiltiy of dropout.
          net_id: int. Id of the network.
        return:
          None
        """
        L1 = []
        L1.extend((
            spektral.layers.MessagePassing(aggregate='sum',
                                          activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout)))
        L2 = []
        L2.extend((
            spektral.layers.MessagePassing(aggregate='sum',
                                           activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout)))
        self._nets[net_id] = (L1, L2)


    def run_MPNN_unit(self, Adj, X, net_id=1):
        """
        Function calls layer given by id
        args:
          Adj: Tensor. [:, x, y]. Adjacency matrix of the graph
          X: Tensor. [:, x, z]. Node feature matrix.
        returns:
          Tensor. [x, z*2]. ouput of passig a message through the MPNN
        """
        L1, L2 = self._nets[net_id]
        y = None
        for i in range(0,len(L1)):
            if i == 0: # MessagePassing layer
                y = L1[i].propagate(X, Adj)
                continue
            y = L1[i](y)
        H1 = y
        for i in range(0, len(L2)):
            if i == 0: # MessagePassing Layer
                y = L2[i].propagate(y, Adj)
                continue
            y = L2[i](y)
        H2 = y
        return tf.concat((H1,H2), axis=1)

    def train(self, model_input, y):
        Adj, X = model_input
        Adj_shape = Adj.shape[0]
        for i in range(0, adj_shape):
            Adj[i] = tf.map_fn(fn=sp_matrix_to_sp_tensor, elems=Adj[i])
        print(Adj.shape)
        return
        with tf.GradientTape() as tape:
            y_pred = self(model_input, training=True)
            loss = tf.keras.losses.mean_squared_error(y, y_pred)

        trainable_vars = self.trainable_vars
        gradients = tape.gradients(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(graidents, trainable_vars))

        # metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}
    
    def call(self, model_input, training=True):
        """
        Function calls the net
        args:
          Adj: Tensor. [:, x, y]. Adjacency matrix of the graph
          X: Tensor. [:, x, z]. Node feature matrix
        returns:
        
        """
        Adj, X = model_input
        print(Adj.shape)
        print(X.shape)
        LSTM_input = []
        for i in range(0, self._window):
            LSTM_input.append(
                self.run_MPNN_unit(Adj[:,i,:,:],
                                   X[:,i,:,:], net_id=i+1))
        """
        H1 = self.run_MPNN_unit(Adj[0], X[0], net_id=1)
        H2 = self.run_MPNN_unit(Adj[1], X[1], net_id=2)
        H3 = self.run_MPNN_unit(Adj[2], X[2], net_id=3)
        H4 = self.run_MPNN_unit(Adj[3], X[3], net_id=4)
        H5 = self.run_MPNN_unit(Adj[4], X[4], net_id=5)
        H6 = self.run_MPNN_unit(Adj[5], X[5], net_id=6)
        LSTM_input = [H1, H2, H3, H4, H5, H6]
        #for i in range(0,len(LSTM_input)):
         #   LSTM_input[i] = tf.expand_dims(LSTM_input[i], axis=0)
        """
        x = tf.stack(LSTM_input, axis=0)
        x = self.LSTM1(x)
        output, final_memory_state, final_carry_state = self.LSTM2(x)
        #x = X+x
        #Lin?
        x = final_memory_state
        x = self.Lin(x)
        return x
