import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing import preprocessing

# preprocessing of data
filename = '../Data/data_trim.csv'
data = preprocessing(filename)
# data = pd.read_csv(filename, usecols=['tweet', 'label'])
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0)
bow = vectorizer.fit_transform(corpus)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)

# AdaBoost
eta = np.logspace(-1, 0, 15)
train_acc = np.zeros(len(eta))
test_acc = np.zeros(len(eta))

for i in range(len(eta)):
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',
                                                        max_depth=1),
                                 n_estimators=100,
                                 algorithm='SAMME.R',
                                 learning_rate=eta[i])
    ada_clf.fit(bow_train, labels_train)

    # predict sentiment of tweetss
    ada_pred_train = ada_clf.predict(bow_train)
    ada_pred_test = ada_clf.predict(bow_test)

    # calculate accuracy score
    test_acc[i] = accuracy_score(labels_test, ada_pred_test)
    train_acc[i] = accuracy_score(labels_train, ada_pred_train)
    print(test_acc)
    print(train_acc)

plt.style.use("ggplot")
plt.plot(eta, test_acc, label='test')
plt.plot(eta, train_acc, label='train')
plt.title('AdaBoost', size=16)
plt.xlabel('Learning rate', size=14)
plt.ylabel('Accuracy score', size=14)
plt.legend()
plt.show()
