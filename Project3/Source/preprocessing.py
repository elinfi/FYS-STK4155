import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def preprocessing(filename):
    df = pd.read_csv(filename, usecols=[1, 2])
    # df = df[:10]
    # df = pd.DataFrame({'tweet': [r":-) :-&gt; :L :S :-O", r"oiveroi:) jovnrs :-*:P", r"kverbx-Doinv joij ;D"], 'labels': [1, 2, 3]})

    # replace url
    url_reg = r"(http(?:s){0,1}://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]"\
              + r"{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
    df = df.replace(to_replace=url_reg, value=' URL ', regex=True)

    # replace @Username
    df = df.replace(to_replace=r"\@[\w_]*", value=' username ', regex=True)

    positive = (r":-\)|:\)|:-\]|:]|:-3|:3|:-&gt;|:&gt;|8-\)|8\)|:-\}|:\}|:o\)|"
                + r":c\)|:\^\)|=\]|=\)|:-D|:D|8-D|8D|x-D|xD|X-D|XD|=D|=3|B\^D"
                + r":'-\)|:'\)|:-\*|:\*|;-\)|;\)|\*-\)|\*\)|;-\]|;\]|;\^\)|:-,|"
                + r";D|&lt;3+")
    negative = (r":-\(|:\(|:-c|:c|:-&lt;|:&lt;|:-\[|:\[|:-\|\||&gt;:\[|:\{|:@|"
                + r";\(|:'-\(|:'\(|D-':|D:&lt;|D:|D8|D;|D=|:-\/|:\/|:-\.|"
                + r"&gt;:\\|&gt;:\/|:\\|=\/|=\\|:L|=L|:S|&gt;:-\)|&gt;:\)|"
                + r"\}:-\)|\}:\)|3:-\)|3:\)|&gt;;\)|&gt;:3|;3|&gt;:\]|"
                + r":-###\.\.|:###\.\.|&lt;\/3+|&lt;\\3+")
    neutral = (r":-O|:O|:-o|:o|:-0|8-0|&gt;|:-\||:\||:\$|:\/\/\)|:\/\/3|:-X|:X|"
               + r":-#|:#|:-&|:&|&lt;:-\|',:-\||',:-l|%-\)|%\)|:E|O_O|o-o|O_o|"
               + r"o_O|o_o|O-O|O\.o|O\.O|o\.o|o\.O")
    delete = r":-P|:P|X-P|XP|x-p|xp|:-p|:p|:-b|:b|d:|=p|&gt;:P"

    # delete tweets with both positive and negative emoticons
    pos_bool = df['tweet'].str.contains(positive, regex=True)
    neg_bool = df['tweet'].str.contains(negative, regex=True)
    pos_neg = pos_bool&neg_bool
    df = df.loc[~pos_neg]

    # delete tweets with variants of :P
    delete_bool = df['tweet'].str.contains(delete, regex=True)
    df = df.loc[~delete_bool]

    # replace postive and negative emoticons
    df = df.replace(to_replace=positive, value=' positive ', regex=True)
    df = df.replace(to_replace=negative, value=' negative ', regex=True)

    # remove neutral emoticons
    df = df.replace(to_replace=neutral, value=' ', regex=True)

    # convert all string to lowercase
    df['tweet'] = df['tweet'].str.lower()

    # remove repeated tweets
    df = df.drop_duplicates()

    # delete retweets
    RT = df['tweet'].str.contains(r"\bRT @|\brt @|\bRt @", regex=True)
    df = df.loc[~RT]


    # replace n't ending of words with not
    df = df.replace(to_replace=r"can't|cannot", value=r"can not", regex=True)
    df = df.replace(to_replace=r"won't", value=r"will not", regex=True)
    df = df.replace(to_replace=r"([a-zA-Z]+)(n't)",
                    value=r"\1 not",
                    regex=True)


    # remove text
    df = df.replace(to_replace=[r"&quot;", # quotation
                                r"&gt;", # greater than >
                                r"&lt;", # less than <
                                r"[^a-zA-Z-' ]", # special characters
                                # r":\)|:-\)|: \)|:D|:-D|=\)|:\]|:\(|:-\(|: \(|:\[|;\)|xD|XD|:P|;P|8D|8\)|o.O|o.o|O.O|O.o",
                                # r"[^a-zA-Z-' ]", # special characters
                                r"\b[a-zA-Z]{1}\b"], # 1 and 2 char words
                    value=r"",
                    regex=True)

    df = df.replace(to_replace=r"&amp;", # ampresand &
                    value=r" ",
                    regex=True)

    df = df.replace(to_replace=r"(xo)(?:\1)+x{0,1}",
                    value=r"xoxo",
                    regex=True)

    df = df.replace(to_replace=r"(ha)(?:\1)+h{0,1}",
                    value=r"haha",
                    regex=True)

    # df = df.replace(to_replace=r"(xo|ha)(?:\1)+[x|h]{0,1}", # xoxoxo...
    #                 value=r"\1",
    #                 regex=True)

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

    df.to_csv('../Data/data_trim_processed.csv')

    return df

if __name__ == '__main__':
    filename = '../Data/data_trim.csv'
    filename2 = '../archive/training.1600000.processed.noemoticon.csv'

    data = preprocessing(filename)
    corpus = data['tweet']
    # labels = data['label']

    print(data)
    # create bag of words
    vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0)
    bow = vectorizer.fit_transform(corpus)
    #
    # print(vectorizer.get_feature_names())
    print(bow.shape)
