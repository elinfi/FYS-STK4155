import re
import nltk
import pandas as pd

from nltk.corpus import stopwords

def preprocessing(filename):
    df = pd.read_csv(filename, usecols=[1, 2])
    test = pd.Series(["oviovnre them she iovne cat ionc", "oviern iosnf doesn't most iocn wasn't", "i STILL have the 2 bricks, that wasn't my stuff lastnight."])

    # remove repeated tweets
    df = df.drop_duplicates()

    # remove retweets
    RT = df['tweet'].str.contains(r"\bRT @|\brt @|\bRt @")
    df = df.loc[~RT]

    # remove tweets containing both sad and smiley face
    smile = df['tweet'].str.contains(r":\)|:-\)|: \)|:D|:-D|=\)|:\]")
    sad = df['tweet'].str.contains(r":\(|:-\(|: \(|:\[")
    both = smile&sad
    df = df.loc[~both]


    # tounge = df['tweet'].str.contains(r":P|:\|")
    # df = df.loc[~tounge]

    # remove text
    url_reg = r"(http(?:s){0,1}://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]"\
              + r"{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
    df = df.replace(to_replace=[r"\@[\w_]*", # @Username
                                url_reg, # URL
                                r"&quot;", # quotation
                                r"&gt;", # greater than >
                                r"[^a-zA-Z:\(\)\[\]=\-' ]", # special characters
                                r"\b[a-zA-Z]{1,2}\b"], # 1 and 2 char words
                    value=r"",
                    regex=True)

    df = df.replace(to_replace=r"&amp;", # ampresand &
                    value=r" ",
                    regex=True)

    # replace repeated characters with to repetitions
    df = df.replace(to_replace=r"([a-zA-Z])(?:\1){2,}",
                    value=r"\1\1",
                    regex=True)

    # remove stopwords
    stopword = r"\b{}\b".format(r'\b|\b'.join(stopwords.words('english')))
    df = df.replace(to_replace=stopword, value=r"", regex=True)
    df = df.replace(to_replace=r"'", value=r"", regex=True)
    # hei = test.replace(to_replace=stopword, value=r"", regex=True)
    # hei = hei.replace(to_replace=r"\b[\']\b", value=r"", regex=True)

    # tokenize data
    tokens = df['tweet'].str.split()

    # remove suffixes from words
    stemmer = nltk.stem.SnowballStemmer("english")
    stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])
    df['tweet'] = stemmed_tokens.str.join(' ')

    # lemma = nltk.wordnet.WordNetLemmatizer()
    # lemmatized_tokens = tokens.apply(lambda x: [lemma.lemmatize(i) for i in x])
    # df['tweet'] = lemmatized_tokens.str.join(' ')

    return df

if __name__ == '__main__':
    filename = '..Data/data_trim.csv'
    filename2 = '..archive/training.1600000.processed.noemoticon.csv'
    print(preprocessing(filename)[120:137])
