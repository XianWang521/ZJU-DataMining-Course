import numpy as np


class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        # the number of labels is n
        dic = {}  
        # calculate the number of samples, simplify the samples
        for i in range(y.shape[0]):
            if y[i] not in dic.keys():
                dic[y[i]] = sample_weights[i]
            else:
                dic[y[i]] += sample_weights[i]
                
        # calculate the entropy of the set of input samples
        for i in dic:
            prop = float(dic[i]) / float(np.sum(sample_weights))
            entropy -= prop * np.log2(prop)
        # end answer
        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0
        # YOUR CODE HERE
        # begin answer
        info_gain = self.entropy(y, sample_weights)
        value_subsets = np.unique(X[:, index])
        
        for i in value_subsets:
            X_subset, y_subset, sample_weights_subset = self._split_dataset(X, y, index, i, sample_weights)
            prop = float(np.sum(sample_weights_subset)) / float(np.sum(sample_weights))
            info_gain -= prop * self.entropy(y_subset, sample_weights_subset)
        # end answer
        return info_gain

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
        """   
        # YOUR CODE HERE
        # This part is similar to calculate the information_gain
        # begin answer
        info_gain = self.entropy(y, sample_weights)
        value_subsets = np.unique(X[:, index])
        split_info = 0.0
        
        for i in value_subsets:
            X_subset, y_subset, sample_weights_subset = self._split_dataset(X, y, index, i, sample_weights)
            prop = float(np.sum(sample_weights_subset)) / float(np.sum(sample_weights))
            info_gain -= prop * self.entropy(y_subset, sample_weights_subset)
            split_info -= prop * np.log2(prop)
        
        if split_info != 0.0:
            info_gain_ratio = info_gain / split_info
        else:
            info_gain_ratio = 0.0
        # end answer
        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        dic = {}
        # calculate the number of samples, simplify the samples
        for i in range(y.shape[0]):
            if y[i] not in dic.keys():
                dic[y[i]] = sample_weights[i]
            else:
                dic[y[i]] += sample_weights[i]
        
        for i in dic:
            prop = float(dic[i]) / float(np.sum(sample_weights))
            gini -= prop * prop
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1
        # YOUR CODE HERE
        # begin answer
        new_impurity = self.gini_impurity(y, sample_weights)
        value_subsets = np.unique(X[:, index])
        
        for i in value_subsets:
            X_subset, y_subset, sample_weights_subset = self._split_dataset(X, y, index, i, sample_weights)
            prop = float(np.sum(sample_weights_subset)) / float(np.sum(sample_weights))
            new_impurity -= prop * self.gini_impurity(y_subset, sample_weights_subset)  
        # end answer
        return new_impurity
    
    # refer https://blog.csdn.net/u010246947/article/details/53258931
    def _split_dataset(self, X, y, index, value, sample_weights):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        n, p = X.shape
        res = []
        temp = X[:, index]
        for i in range(n):
            if temp[i] == value:
                res.append(i)
        
        sub_y = y[res]
        sub_sample_weights = sample_weights[res]
        X = X[:, [i for i in range(p) if i != index]]
        sub_X = X[res, :]
        # end answer
        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        n, p = X.shape
        best = 0
        # because here is classification, so the size of feature subsets is sqrt(total number of features)
        if self.sample_feature:
            num_features = int(np.round(np.sqrt(p)))
            features = np.random.choice(p, num_features, replace = False)
            new = X[:, features]
        else:
            new = X
        
        new_n, new_p = new.shape
        for i in range(new_p):
            temp = self.criterion(new, y, i, sample_weights)
            if temp > best:
                best = temp
                best_feature_idx = i
        # end answer
        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        dic = {}
        for i in range(y.shape[0]):
            if y[i] not in dic.keys():
                dic[y[i]] = sample_weights[i]
            else:
                dic[y[i]] += sample_weights[i]
        
        majority_label = max(dic, key = dic.get)
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf, we use the majority_label by calling majority_vote.
                mytree = {
                    'titile': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X. Then use `_split_dataset` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful.
        # begin answer
        if depth >= self.max_depth or X.shape[0] <= self.min_samples_leaf or len(feature_names) == 0 or np.unique(y).shape[0] == 1:
            return self.majority_vote(y, sample_weights)
        
        best_feature_idx = self._choose_best_feature(X, y, sample_weights)
        mytree = {feature_names[best_feature_idx]:{}}
        value_subsets = np.unique(X[:, best_feature_idx])
        temp = feature_names[best_feature_idx]
        feature_names = [i for i in feature_names if i != temp]
        
        for i in value_subsets:
            X_subset, y_subset, sample_weights_subset = self._split_dataset(X, y, best_feature_idx, i, sample_weights)
            mytree[temp][i] = self._build_tree(X_subset, y_subset, feature_names, depth + 1, sample_weights_subset)
        # end answer
        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            fatherFeature = list(tree.keys())[0]
            childFeature = tree[fatherFeature]
            key = x.loc[fatherFeature]
            # randomly choose a subtree if the feature value in testing data not exists in training data.
            if key not in childFeature:
                key = np.random.choice(list(childFeature.keys()))
            valueOfFea = childFeature[key]
            
            if isinstance(valueOfFea, dict):
                label = _classify(valueOfFea, x)
            else:
                label = valueOfFea
            return label
            # end answer

        # YOUR CODE HERE
        # begin answer
        res = []
        for i in range(X.shape[0]):
            temp = _classify(self._tree, X.iloc[i, :])
            res.append(temp)
        return np.array(res)
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
