import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')

X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
    
# Add Polynomial Features of degree 2
def addPolynomialFeatures(X, imp) :
    for s in imp :
        X[s + '2'] = X[s]**2
    
    X[' INDUS_RM'] = X[' INDUS']*X[' RM']
    X[' INDUS_LSTAT'] = X[' INDUS']*X[' LSTAT']
    X[' RM_NOX'] = X[' RM']*X[' NOX']
    X[' CRIM_INDUS'] = X['# CRIM']*X[' INDUS']
    X[' INDUS_AGE'] = X[' INDUS'] * X[' AGE']
    X[' RM_AGE'] = X[' RM'] * X[' AGE']
    return X
    
# Add a new Bias Column containing all ones
def addBiasColumn(X) :
    X["bias"] = np.ones(len(X.values))
    return X
    
# Predict cost if the chosen slopes are m
def cost(X, Y, m) :
    M = len(X.values)
    N = len(X.values[0])
    
    total_cost = 0
    
    for i in range(M) :
        predict = 0
        
        for j in range(N) :
            predict += m[j]*X.values[i, j]
        
        total_cost += ((Y.values[i] - predict)**2)
        
    total_cost /= M
    
    return total_cost
       
# Perform gradient descent 
def gradient_descent(lr, epochs) :
    N = len(X.values[0])
    M = len(X.values)
    m = np.ones(N)
    for e in range(epochs) :
        m_copy = np.ones(N)
        print(e, ' ', cost(X, Y, m))
        for i in range(N) :
            m_copy[i] = m[i]
        for k in range(N) :
            term = 0
            for i in range(M) :
                y_pred = 0
                for j in range(N) :
                    y_pred += m_copy[j]*X.values[i, j]
                term -= 2*(Y.values[i] - y_pred)*X.values[i, k]
            term /= M
            m[k] = m[k] - (lr*term)
    return m

def predict(X, m) :
    M = len(X.values)
    N = len(X.values[0])
    
    y_pred = np.ones(M)
    
    for i in range(M) :
        y = 0
        for j in range(N) :
            y += m[j]*X.values[i, j]
        y_pred[i] = y
    return y_pred
    
col = X.columns
X = addBiasColumn(X)
imp = [' INDUS', ' RM', ' LSTAT', ' NOX', ' AGE']

print(X.columns)       
       
X = addPolynomialFeatures(X, imp)

m = gradient_descent(0.08, 400)

x_test = pd.DataFrame(np.loadtxt('test.csv', delimiter = ','))
x_test.columns = col
x_test = addBiasColumn(x_test)
x_test = addPolynomialFeatures(x_test, imp)
y_predict = predict(x_test, m)

np.savetxt('sol.csv', y_predict, fmt = '%.5f')