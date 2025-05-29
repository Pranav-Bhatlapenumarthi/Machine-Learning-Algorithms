import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = None  # For leaf nodes, this will hold the class label or regression value

    def is_leaf_node(self): # To check if the node is a leaf node
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else self.num_features
        self.root = self.grow_tree(X, y) # growing the decision tree
    
    def grow_tree(self, X, y):
        num_samples, num_features = X.shape()
        num_labels = len(np.unique(y))

        #checking the stopping criteria
        if(depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        feat_index = np.random.choice(num_features, self.num_features, replace=True) # To randomly select features

        # find the best split
        best_threshold, best_feature = self._best_split(X, y, feat_index)


        # create child nodes





    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
