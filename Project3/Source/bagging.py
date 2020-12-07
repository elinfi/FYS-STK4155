import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing import preprocessing

# preprocessing of data
filename = '../Data/data_trim.csv'
data = preprocessing(filename)
data = data[:10000]
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=0.001, max_df=0.9999)
bow = vectorizer.fit_transform(corpus)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)
# perform bagging on decision trees
bag_clf = BaggingClassifier(DecisionTreeClassifier(criterion='gini',
                                                   max_depth=100),
                            n_estimators=500,
                            max_samples=1.0,
                            bootstrap=True,
                            n_jobs=-1,
                            # max_features=1000,
                            # bootstrap_features=True,
                            oob_score=True)
bag_clf.fit(bow_train, labels_train)
print(bag_clf.oob_score_)

# predict sentiment of tweetss
bag_pred_train = bag_clf.predict(bow_train)
bag_pred_test = bag_clf.predict(bow_test)

# calculate accuracy score
print(accuracy_score(labels_test, bag_pred_test))
print(accuracy_score(labels_train, bag_pred_train))
