import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocessing

# preprocessign of data
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

estimiator_list = np.logspace(0, 2.3, 20)
test_acc = np.zeros(len(estimiator_list))
train_acc = np.zeros(len(estimiator_list))

for i in range(len(estimiator_list)):
    # create random forest
    rnd_clf = RandomForestClassifier(n_estimators=int(estimiator_list[i]),
                                     max_depth=100,
                                     n_jobs=-1,
                                     random_state=4731)
    rnd_clf.fit(bow_train, labels_train)

    # predict sentiment of tweets
    rnd_test = rnd_clf.predict(bow_test)
    rnd_train = rnd_clf.predict(bow_train)

    # calculate accuracy score
    test_acc[i] = accuracy_score(labels_test, rnd_test)
    train_acc[i] = accuracy_score(labels_train, rnd_train)

    print(test_acc)
    print(train_acc)

plt.style.use("ggplot")
plt.plot(estimiator_list, test_acc, label='test')
plt.plot(estimiator_list, train_acc, label='train')
plt.title('Random forest', size=16)
plt.xlabel('Number of estimators', size=14)
plt.ylabel('Accuracy score', size=14)
plt.legend()
plt.show()
