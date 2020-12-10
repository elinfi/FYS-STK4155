import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing import preprocessing

# preprocessign of data
filename = '../Data/data_trim.csv'
# data = preprocessing(filename)
data = pd.read_csv(filename)
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=5, max_df=0.99)
bow = vectorizer.fit_transform(corpus)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)

# create random forest
rnd_clf = RandomForestClassifier(n_estimators=100, max_depth=100, n_jobs=-1)
rnd_clf.fit(bow_train, labels_train)

# predict sentiment of tweets
rnd_test = rnd_clf.predict(bow_test)
rnd_train = rnd_clf.predict(bow_train)

# calculate accuracy score
print(accuracy_score(labels_test, rnd_test))
print(accuracy_score(labels_train, rnd_train))
