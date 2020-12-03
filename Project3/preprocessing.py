import re
import nltk
import pandas as pd

from nltk.corpus import stopwords

def preprocessing(filename):
    df = pd.read_csv(filename, usecols=[1, 2])
    test = pd.Series(["oviovnre them she iovne cat ionc", "oviern iosnf most iocn wasn't"])

    # remove retweets
    RT = df['tweet'].str.contains(r" RT @| rt @| Rt @")
    df = df.loc[~RT]
    RT = df['tweet'].str.match(r"RT @|rt @|Rt @")
    df = df.loc[~RT]

    # remove tweets containing both sad and smiley face
    smile = df['tweet'].str.contains(r":\)|:-\)|: \)|:D|=\)|:\]")
    sad = df['tweet'].str.contains(r":\(|:-\(|: \(|:\[")
    both = smile&sad
    df = df.loc[~both]

    # remove repeated tweets
    df = df.drop_duplicates()

    # tounge = df['tweet'].str.contains(r":P|:\|")
    # df = df.loc[~tounge]

    # remove text
    df = df.replace(to_replace=[r"\@[\w_]*", # @Username
                                r"http[s]*:\/\/[^\s]*", # URL
                                r"&quot;", # quotation
                                r"[^a-zA-Z:\(\)\[\]=\-' ]", # special characters
                                r"\b[a-zA-Z]{1,2}\b"], # 1 and 2 char words
                    value=r"",
                    regex=True)

    # replace repeated characters with to repetitions
    df = df.replace(to_replace=r"([a-zA-Z])(?:\1){2,}",
                    value=r"\1\1",
                    regex=True)

    # remove stopwords
    # print(type(stopwords.words('english')))
    stopword =  [r"\bi\b|\bme\b|\bmy\b|\bmyself\b|\bwe\b|\bour\b|\bours\b|"
                  + r"\bourselves\b|\byou\b|\byou're\b|\byou've\b|\byou'll\b|"
                  + r"\byou'd\b|\byour\b|\byours\b|\byourself\b|\byourselves\b|"
                  + r"\bhe\b|\bhim\b|\bhis\b|\bhimself\b|\bshe\b|\bshe's\b|"
                  + r"\bher\b|\bhers\b|\bherself\b|\bit\b|\bit's\b|\bits\b|"
                  + r"\bitself\b|\bthey\b|\bthem\b|\btheir\b|\btheirs\b|"
                  + r"\bthemselves\b|\bwhat\b|\bwhich\b|\bwho\b|\bwhom\b|"
                  + r"\bthis\b|\bthat\b|\bthat'll\b|\bthese\b|\bthose\b|\bam\b|"
                  + r"\bis\b|\bare\b|\bwas\b|\bwere\b|\bbe\b|\bbeen\b|"
                  + r"\bbeing\b|\bhave\b|\bhas\b|\bhad\b|\bhaving\b|\bdo\b|"
                  + r"\bdoes\b|\bdid\b|\bdoing\b|\ba\b|\ban\b|\bthe\b|\band\b|"
                  + r"\bbut\b|\bif\b|\bor\b|\bbecause\b|\bas\b|\buntil\b|"
                  + r"\bwhile\b|\bof\b|\bat\b|\bby\b|\bfor\b|\bwith\b|"
                  + r"\babout\b|\bagainst\b|\bbetween\b|\binto\b|\bthrough\b|"
                  + r"\bduring\b|\bbefore\b|\bafter\b|\babove\b|\bbelow\b|"
                  + r"\bto\b|\bfrom\b|\bup\b|\bdown\b|\bin\b|\bout\b|\bon\b|"
                  + r"\boff\b|\bover\b|\bunder\b|\bagain\b|\bfurther\b|"
                  + r"\bthen\b|\bonce\b|\bhere\b|\bthere\b|\bwhen\b|\bwhere\b|"
                  + r"\bwhy\b|\bhow\b|\ball\b|\bany\b|\bboth\b|\beach\b|"
                  + r"\bfew\b|\bmore\b|\bmost\b|\bother\b|\bsome\b|\bsuch\b|"
                  + r"\bno\b|\bnor\b|\bnot\b|\bonly\b|\bown\b|\bsame\b|\bso\b|"
                  + r"\bthan\b|\btoo\b|\bvery\b|\bs\b|\bt\b|\bcan\b|\bwill\b|"
                  + r"\bjust\b|\bdon\b|\bdon't\b|\bshould\b|\bshould've\b|"
                  + r"\bnow\b|\bd\b|\bll\b|\bm\b|\bo\b|\bre\b|\bve\b|\by\b|"
                  + r"\bain\b|\baren\b|\baren't\b|\bcouldn\b|\bcouldn't\b|"
                  + r"\bdidn\b|\bdidn't\b|\bdoesn\b|\bdoesn't\b|\bhadn\b|"
                  + r"\bhadn't\b|\bhasn\b|\bhasn't\b|\bhaven\b|\bhaven't\b|"
                  + r"\bisn\b|\bisn't\b|\bma\b|\bmightn\b|\bmightn't\b|"
                  + r"\bmustn\b|\bmustn't\b|\bneedn\b|\bneedn't\b|\bshan\b|"
                  + r"\bshan't\b|\bshouldn\b|\bshouldn't\b|\bwasn\b|\bwasn't\b|"
                  + r"\bweren\b|\bweren't\b|\bwon\b|\bwon't\b|\bwouldn\b|"
                  + r"\bwouldn't\b"]

    df = df.replace(to_replace=stopword,
                        value=r"",
                        regex=True)
    hei = test.replace(to_replace=stopword, value=r"", regex=True)

    # tokenize data
    tokens = df['tweet'].str.split()

    # remove suffixes from words
    stemmer = nltk.stem.SnowballStemmer("english")
    stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])
    df['tweet'] = stemmed_tokens.str.join(' ')

    # lemma = nltk.wordnet.WordNetLemmatizer()
    # lemmatized_tokens = tokens.apply(lambda x: [lemma.lemmatize(i) for i in x])
    # df['tweet'] = lemmatized_tokens.str.join(' ')

    return hei

if __name__ == '__main__':
    filename = 'data_trim.csv'
    filename2 = 'archive/training.1600000.processed.noemoticon.csv'
    print(preprocessing(filename))
