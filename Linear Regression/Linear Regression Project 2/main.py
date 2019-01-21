# Importing Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Import Training Data
train = pd.read_csv('train.csv')
X = train.iloc[:, :-1]
Y = train.iloc[:, -1]

# Add Additional Columns here
X['V_V'] = X[' V']**2
X['RH_RH'] = X[' RH']**2

columns = X.columns

# Feature Scaling
scaler = preprocessing.StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = columns

# Adding Bias Column
X['bias'] = np.ones(len(X.values))

# Cost function
def cost(m) :
    M = len(X.values)
    N = len(X.values[0])
    
    total_cost = 0
    
    for i in range(M) :
        y_pred = 0
        for j in range(N) :
            y_pred += m[j]*X.values[i, j]
        total_cost += ((Y[i] - y_pred)**2)
    
    total_cost /= M
    return total_cost

# Gradient Descent
def gradient_descent(lr, epochs) :
    M = len(X.values)
    N = len(X.values[0])
    
    m = np.ones(N)
    
    for e in range(epochs) :
        print(e, ' ', cost(m))
        m_copy = np.ones(N)
        y_pred = np.ones(M)
        for i in range(N) :
            m_copy[i] = m[i]
        for i in range(M) :
            y = 0
            for j in range(N) :
                y += m_copy[j]*X.values[i, j]
            y_pred[i] = y
        for k in range(N) :
            term = 0
            for i in range(M) :
                term -= (2*(Y[i] - y_pred[i])*X.values[i, k])
            term /= M
            m[k] -= (lr*term)
    return m

m = gradient_descent(0.1, 400)

# Import Testing Data
X_test = pd.DataFrame(np.loadtxt('test.csv', delimiter=','))

# Add additional columns
X_test['V_V'] = X_test[1]**2
X_test['RH_RH'] = X_test[3]**2

# Feature Scaling
X_test = pd.DataFrame(scaler.transform(X_test))

# Bias Column
X_test["bias"] = np.ones(len(X_test.values))

X_test.columns = X.columns

# Predict
def predict() :
    N = len(X_test.values[0])
    M = len(X_test.values)
    
    y_pred = np.ones(M)
    
    for i in range(M) :
        y = 0
        for j in range(N) :
            y += m[j]*X_test.values[i, j]
        y_pred[i] = y
    
    return y_pred

y_pred = predict()

# Store Results
np.savetxt('sol.csv', y_pred, fmt = '%.5f')