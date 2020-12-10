import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def preprocessing(filename):
    df = pd.read_csv(filename, usecols=['label', 'tweet'])

    # delete retweets
    RT = df['tweet'].str.contains(r"\bRT @|\brt @|\bRt @", regex=True)
    df = df.loc[~RT]

    # remove repeated tweets
    df = df.drop_duplicates()

    # replace url
    url_reg = r"(http(?:s){0,1}://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]"\
              + r"{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
    df = df.replace(to_replace=url_reg, value=' url ', regex=True)

    # replace @Username
    df = df.replace(to_replace=r"@[\w_]{1,15}\s", value=' username ',
                    regex=True)

    # regex for emoticons
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

    # remove text
    df = df.replace(to_replace=[r"&quot;", # quotation
                                r"&gt;", # greater than >
                                r"&lt;", # less than <
                                r"&amp;", # ampresand &
                                r"[^a-zA-Z-' ]"], # special characters
                    value=r" ",
                    regex=True)

    # replace repetitions of xoxo and haha
    df = df.replace(to_replace=r"(xo)(?:\1)+x{0,1}",
                    value=r"xoxo",
                    regex=True)
    df = df.replace(to_replace=r"(ha)(?:\1)+h{0,1}",
                    value=r"haha",
                    regex=True)

    # replace repeated characters with two repetitions
    df = df.replace(to_replace=r"([a-zA-Z])(?:\1){2,}",
                    value=r"\1\1",
                    regex=True)

    # replace negations with not
    negation_list = ['ain', 'aint', 'aren', 'arent', "aren't", 'couldn',
                     'couldnt', "couldn't", 'didn', 'didnt', "didn't", 'doesn',
                     'doesnt', "doesn't", 'hadn', 'hadnt', "hadn't", 'hasn',
                     'hasnt', "hasn't", 'haven', 'havent', "haven't", 'isn',
                     'isnt', "isn't", 'mightn', "mightn't", 'mightnt', 'mustn',
                     'mustnt', "mustn't", 'needn', 'neednt', "needn't", 'shan',
                     'shant', "shan't", 'shouldn', 'shouldnt', "shouldn't",
                     'wasn', 'wasnt', "wasn't", 'weren', 'werent', "weren't",
                     'won', 'wont', "won't", 'wouldn', 'wouldnt', "wouldn't",
                     'don', 'dont', "don't", 'cant', "can't", 'cannot',
                     'darent', "daren't"]
    negation_regex = r"\b{}\b".format(r'\b|\b'.join(negation_list))
    df = df.replace(to_replace=negation_regex, value=r"not", regex=True)
    df = df.replace(to_replace=r"[a-zA-Z]+n't", value=r"not", regex=True)

    # remove stopwords
    stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                     'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                     'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                     'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                     'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                     'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                     'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                     'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                     'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                     'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about',
                     'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up',
                     'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                     'more', 'most', 'other', 'some', 'such', 'only', 'own',
                     'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                     'will', 'just', 'should', "should've", 'now', 'd', 'll',
                     'm', 'o', 're', 've', 'y', 'ma', 'could', 'need']
    stopword_regex = r"\b{}\b".format(r'\b|\b'.join(stopword_list))
    df = df.replace(to_replace=stopword_regex, value=r"", regex=True)

    # remove all apostrophe
    df = df.replace(to_replace=r"'", value=r"", regex=True)

    # tokenize data
    tokens = df['tweet'].str.split()

    # remove suffixes from words
    stemmer = nltk.stem.SnowballStemmer("english")
    stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])

    # join data
    df['tweet'] = stemmed_tokens.str.join(' ')

    # lemma = nltk.wordnet.WordNetLemmatizer()
    # lemmatized_tokens = tokens.apply(lambda x: [lemma.lemmatize(i) for i in x])
    # df['tweet'] = lemmatized_tokens.str.join(' ')

    # write preprocessed data to file
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
