# Source code

`preprocessing.py`

- Code for preprocessing of data. Takes as input a csv-file containing tweets with associated sentiment labeling. The preprocessing is specifically created to handle tweets.

`decision_tree.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Creates a decision tree using gini score for splitting and prints out the corresponding accuracy score for train and test data.

`bagging.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Performs bagging on the bag of words and prints the corresponding accuracy score for test and train data.

`random_forest.py`

- Creates a bag of words using TF-IDF vectorizer on eiter preprocessed data or original data. Creates a random forest and prints out the corresponding accuracy score for train and test data.s

`ada_boost.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Performs AdaBoost on decision stumps for a given list of learning rate values. Prints and plots the corresponding accuracy score for both train and test data.

`xg_boost.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Performs XGBoost on decision trees and prints the corresponding accuracy score for train and test data.
