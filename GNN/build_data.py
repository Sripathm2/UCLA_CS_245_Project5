import pandas as pd
import numpy as np

def _build_data(data, days):
    """
    Function builds the adj matrix and inital featuer matrix for 
    input in MPNN network
    args:
      data: pandas DataFrame. Contains new cases data for all states
      days: Int. The number of days stored be each node
    Returns:
      Adj. Numpy array of shape [x, 51, 51]. Adjacency matrix for the graph
      X: Numpy array of shape [x, 51, days]. Nodes in graph store new cases for n days.
      y: Numpy array of shape [x, 1]. Stores new cases for every 'days+1'.
    """
    y = data.iloc[days:,:].values
    X = [data.iloc[i:i+days,:].values.T
         for i in range(0,data.shape[0]-days,1)]
    states = [i[1] for i in data.columns.values]
    Adj = _build_ADJ_Mat(X, states)
    X = np.dstack(X)
    X = np.rollaxis(X, 2, 0)
    y = np.expand_dims(y,2)
    return Adj, X, y

def _build_ADJ_Mat(data_list, states):
    """
    Function builds the adjacency matrix for the given values
    data_list: List of numpy arrays. Data used to construct the adjacency matrix.
    states: List of string. List of the states in correct order for Adj matrix
    Returns:
      adj. Numpy array of shape [x, 51, 51]. x is the amount of data. 51 states
    """
    bstates = pd.read_csv("../Clean_Data/bstates.csv").values
    states_idx = {states[i]:i for i in range(0,len(states))}
    statePops = pd.read_csv("../Clean_Data/State_Pops.csv").values.tolist()
    statePops = {a:b for a,b in statePops}
    bstates = {i[0]:[x for x in i[1:] if x==x] for i in bstates}
    adj = []
    for data in data_list:
        mat = np.zeros((51,51))
        average = np.mean(data, axis=1)
        for state in bstates:
            for bstate in bstates[state]:
                mat[states_idx[state]][states_idx[bstate]] = (
                    average[states_idx[state]]/statePops[state])-(
                        average[states_idx[bstate]]/statePops[bstate])
        adj.append(mat)
    adj = np.dstack(adj)
    adj = np.rollaxis(adj, 2, 0)
    return adj

def build_data():
    data = pd.read_csv("../Clean_Data/Clean_CDC.csv")
    data.State = data.State.replace('NYC', 'NY')
    NYData = data[data.State == 'NY'].groupby('Date', as_index=False).sum()
    data = data[data.State != 'NY']
    data = data[~data.State.isin(["FSG", "FSM", "RMI", "AS", "PR", "GU", "MP", "PW", "VI"])]
    data = pd.concat((data, NYData)).replace(np.nan, 'NY')
    data = data.pivot(index='Date', columns='State', values=['New_cases'])
    Adj, X, y = _build_data(data, 5)
    return Adj, X, y

if __name__ == "__main__":
    Adj, X, y = build_data()
