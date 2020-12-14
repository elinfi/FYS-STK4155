import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocessing

# preprocessing of data
filename = '../Data/data_trim.csv'
data = preprocessing(filename)
# data = pd.read_csv(filename, usecols=['label', 'tweet'])
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
bow = vectorizer.fit_transform(corpus)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)
# perform bagging on decision trees
estimators = [10, 50, 100, 150]
max_depths = [10, 50, 100, 150]

accuracy_heatmap_test = np.zeros((len(estimators), len(max_depths)))
accuracy_heatmap_train = np.zeros((len(estimators), len(max_depths)))

for i, estimator in enumerate(estimators):
    for j, depth in enumerate(max_depths):
        bag_clf = BaggingClassifier(DecisionTreeClassifier(criterion='gini',
                                    max_depth=depth),
                                    n_estimators=estimator,
                                    max_samples=1.0,
                                    bootstrap=True,
                                    n_jobs=-1)
        bag_clf.fit(bow_train, labels_train)

        # predict sentiment of tweetss
        bag_pred_train = bag_clf.predict(bow_train)
        bag_pred_test = bag_clf.predict(bow_test)

        # calculate accuracy score
        test_acc = accuracy_score(labels_test, bag_pred_test)
        train_acc = accuracy_score(labels_train, bag_pred_train)
        accuracy_heatmap_test[i, j] = test_acc
        accuracy_heatmap_train[i, j] = train_acc
        print(i, j)


heatmap = sb.heatmap(accuracy_heatmap_test , annot=True, cmap='YlGnBu_r',
                     xticklabels=estimators, yticklabels=max_depths,
                     cbar_kws={'label': 'Accuracy'})
heatmap.set_xlabel('Number of estimators', size=12)
heatmap.set_ylabel('Max depth', size=12)
heatmap.set_title('Bagging test', size=16)
plt.show()

heatmap = sb.heatmap(accuracy_heatmap_train , annot=True, cmap='YlGnBu_r',
                     xticklabels=estimators, yticklabels=max_depths,
                     cbar_kws={'label': 'Accuracy'})
heatmap.set_xlabel('Number of estimators', size=12)
heatmap.set_ylabel('Max depth', size=12)
heatmap.set_title('Bagging train', size=16)
plt.show()
