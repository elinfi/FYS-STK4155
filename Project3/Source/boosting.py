import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing import preprocessing

# preprocessing of data
filename = '../Data/data_trim.csv'
data = preprocessing(filename)
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=5, max_df=0.99)
bow = vectorizer.fit_transform(corpus)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)

# AdaBoost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',
                                                    max_depth=100),
                             n_estimators=100,
                             algorithm='SAMME.R',
                             learning_rate=0.1)
ada_clf.fit(bow_train, labels_train)

# predict sentiment of tweetss
ada_pred_train = ada_clf.predict(bow_train)
ada_pred_test = ada_clf.predict(bow_test)

# calculate accuracy score
print(accuracy_score(labels_test, ada_pred_test))
print(accuracy_score(labels_train, ada_pred_train))
