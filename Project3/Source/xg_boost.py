import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocessing

# preprocessing of data
filename = '../Data/data_trim.csv'
data = preprocessing(filename)
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0)
bow = vectorizer.fit_transform(corpus)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)

xg_clf = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=100)
xg_clf.fit(bow_train, labels_train)

xg_pred_train = xg_clf.predict(bow_train)
xg_pred_test = xg_clf.predict(bow_test)

print(accuracy_score(labels_test, xg_pred_test))
print(accuracy_score(labels_train, xg_pred_train))
