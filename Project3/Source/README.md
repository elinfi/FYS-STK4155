# Source code

`preprocessing.py`

- Code for preprocessing of data. Takes as input a csv-file containing tweets with associated sentiment labeling. The preprocessing is specifically created to handle tweets.

`decision_tree.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Creates decision trees for varying maximum depth using gini score for splitting and plots out the corresponding accuracy score for train and test data.

`bagging.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Performs bagging on the bag of words for varying learning rate and max depth of the decision trees. Plots corresponding accuracy score for test and train data in heatmaps.

`random_forest.py`

- Creates a bag of words using TF-IDF vectorizer on eiter preprocessed data or original data. Creates random forests for varying number of estimators and plots the corresponding accuracy score for train and test data.

`ada_boost.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Performs AdaBoost on decision stumps for varying learning rate. Plots the corresponding accuracy score for both train and test data.

`xg_boost.py`

- Creates a bag of words using TF-IDF vectorizer on either preprocessed data or original data. Performs XGBoost on decision trees and prints the corresponding accuracy score for train and test data.
