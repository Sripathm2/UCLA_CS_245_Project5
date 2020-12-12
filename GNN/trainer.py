from network import *
from build_data import *

feature_window = 6
network_window = 6
data = build_data(feature_window, network_window)
print("ADJ: ", data[0].shape)
print("X: ", data[1].shape)
print("Y: ", data[2].shape)
# Reform the data for 6 days
Adj = data[0]
X = data[1]
y = data[2]
"""
exit()
Adj = []
X = []
y = []
for i in range(0, data[0].shape[0]-window, 1):
    Adj.append(data[0][i:i+window,:,:])
    X.append(data[1][i:i+window,:])
exit()
size = Adj.shape[0]
batch_size = 24
print(batch_size, size)
"""
model = Net()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics=['loss']
    )

model.train([Adj, X], y)
#model.fit(x=[Adj, X], y=y, batch_size=24)


for i in range(0, size, batch_size):
    B_Adj = Adj[i:i+batch_size, :, :]
    B_X = X[i:i+batch_size, :]
    B_y = X[i+i+batch_size, :]
    B_input = [B_Adj, B_x]
    model.fit(B_input, B_y)
    

# Test data
Adj = np.zeros((52,52))
Adj[5][5] = 1
Adj[3][5] = 2
Adj = sp_matrix_to_sp_tensor(Adj)
ad = [Adj for i in range(0,6)]
X = [np.random.rand(52, 6) for i in range(0,6)]
y = np.random.rand(52, 1)





