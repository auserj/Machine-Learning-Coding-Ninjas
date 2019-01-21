import numpy as np

def fit(X_train, Y_train) :
    result = {}
    class_values = set(Y_train)
    for current_class in class_values :
        result[current_class] = {}
        result["total_data"] = len(Y_train)
        current_class_rows = (Y_train==current_class)
        X_train_current = X_train[current_class_rows]
        Y_train_current = Y_train[current_class_rows]
        num_features = X_train.shape[1]
        result[current_class]['total_count'] = len(Y_train_current)
        for j in range(1, num_features + 1) :
            result[current_class][j] = {}
            all_possible_values = set(X_train[:, j])
            for current_value in all_possible_values:
                result[current_class][j][current_value] = (X_train_current[:, j] == current_value).sum()
                
