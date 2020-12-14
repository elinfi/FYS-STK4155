import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocessing

# preprocessing of data
filename = '../Data/data_trim.csv'
data = preprocessing(filename)
# data = pd.read_csv(filename)
corpus = data['tweet']
labels = data['label']

# create bag of words
vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0)
bow = vectorizer.fit_transform(corpus)
# print(bow)
print(bow.shape)

# split in train and test data
bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                  test_size=0.3)

# create decision tree
list = np.logspace(0, 2.3, 20)
test_acc = np.zeros(len(list))
train_acc = np.zeros(len(list))

for i in range(len(list)):
    dct = DecisionTreeClassifier(criterion='gini', max_depth=list[i],
                                 random_state=4731)
    dct.fit(bow_train, labels_train)

    # predict sentiment of tweets
    dct_pred_test = dct.predict(bow_test)
    dct_pred_train = dct.predict(bow_train)

    # calculate accuracy score
    test_acc[i] = accuracy_score(labels_test, dct_pred_test)
    train_acc[i] = accuracy_score(labels_train, dct_pred_train)

    print(test_acc)
    print(train_acc)

plt.style.use("ggplot")
plt.plot(list, test_acc, label='test')
plt.plot(list, train_acc, label='train')
plt.title('Decision trees', size=16)
plt.xlabel('Max depth', size=14)
plt.ylabel('Accuracy score', size=14)
plt.legend()
plt.show()
