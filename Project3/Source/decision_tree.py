import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import preprocessing

filename = 'data_trim.csv'
data = preprocessing(filename)
corpus = data['tweet']
labels = data['label']

vectorizer = CountVectorizer(min_df=5, max_df=0.99)
bow = vectorizer.fit_transform(corpus)

bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels, test_size=0.3)

print(bow_test.shape)
print(bow_train.shape)

dct = DecisionTreeClassifier(criterion='gini', max_depth=40)
dct.fit(bow_train, labels_train)

dct_bow_test = dct.predict(bow_test)
dct_bow_train = dct.predict(bow_train)

print(dct_bow_test.shape)
print(dct_bow_test)
print(dct_bow_train)
print(accuracy_score(labels_test, dct_bow_test))
print(accuracy_score(labels_train, dct_bow_train))
# df_bow = pd.DataFrame(bow.todense())

# print(df_bow)
