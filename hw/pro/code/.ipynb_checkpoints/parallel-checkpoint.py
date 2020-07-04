from decision_tree import DecisionTree
import numpy as np
def fit_one_tree(X, y):
    dt = DecisionTree(criterion='entropy', max_depth=4, min_samples_leaf=2, sample_feature=True)
    idx = np.random.choice(X.shape[0], X.shape[0], replace = True)
    dt.fit(X.iloc[idx, :], y.iloc[idx])
    return dt.predict(X)

def temp(Z):
    return fit_one_tree(Z[0],Z[1])