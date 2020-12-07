import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
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
# print(bow)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)

# create decision tree
dct = DecisionTreeClassifier(criterion='gini', max_depth=500)
dct.fit(bow_train, labels_train)

# predict sentiment of tweets
dct_pred_test = dct.predict(bow_test)
dct_pred_train = dct.predict(bow_train)

# calculate accuracy score
print(accuracy_score(labels_test, dct_pred_test))
print(accuracy_score(labels_train, dct_pred_train))
# df_bow = pd.DataFrame(bow.todense())

# print(df_bow)
