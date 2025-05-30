import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes, this will hold the class label or regression value

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
    
    def grow_tree(self, X, y, depth = 0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        #checking the stopping criteria
        if(depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        feat_index = np.random.choice(num_features, self.num_features, replace=True) # To randomly select features

        # find the best split
        best_threshold, best_feature = self._best_split(X, y, feat_index)


        # create child nodes
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left = self.grow_tree(X[left_idx, :], y[left_idx], depth+1)
        right = self.grow_tree(X[right_idx, :], y[right_idx], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_index):
        best_gain = -1
        split_index = None
        split_threshold = None

        for feat_idx in feat_index:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                # calculate the information gain
                gain = self._info_gain(y, X_col, thr)

                if gain > best_gain:
                    best_gain = gain 
                    split_index = feat_idx
                    split_threshold = thr

        return split_threshold, split_index
                
    def _info_gain(self, y, X_col, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create child nodes
        left_index, right_index = self._split(X_col, threshold)
        if len(left_index) == 0 or len(right_index) == 0:
            return 0
        
        # calculate the weighted entropy of child nodes
        n = len(y)
        n_left, n_right = len(left_index), len(right_index)
        entropy_left, entropy_right = self._entropy(y[left_index]), self._entropy(y[right_index])
        child_entropy = (n_left/n)*entropy_left + (n_right/n)*entropy_right

        # calculate IG
        IG = parent_entropy - child_entropy
        return IG

    def _split(self, X_col, split_threshold):
        left_idxs = np.argwhere(X_col <= split_threshold).flatten()
        right_idxs = np.argwhere(X_col > split_threshold).flatten()
        return left_idxs, right_idxs


    def _entropy(self, y):
        hist = np.bincount(y)
        p_x = hist/len(y)
        return -np.sum([p*np.log(p) for p in p_x if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)


if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

    clf = DecisionTree()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)


    def accuracy_metric(y_test, y_pred):
        return np.sum(y_test == y_pred)/len(y_test)
    
    acc = accuracy_metric(y_test, predictions)
    print("Accuracy: ", acc)