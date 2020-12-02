import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'data_trim.csv'
df = pd.read_csv(filename, usecols=[1, 2])

# remove retweets
RT = df['tweet'].str.contains(r" RT @| rt @| Rt @")
df = df.loc[~RT]
RT = df['tweet'].str.match(r"RT @|rt @|Rt @")
df = df.loc[~RT]

# remove tweets containing both sad and smiley face
smile = df['tweet'].str.contains(r":\)|:-\)|: \)|:D|=\)")
sad = df['tweet'].str.contains(r":\(|:-\(|: \(")
both = smile&sad
df = df.loc[~both]
print(df)

# remove repeated tweets
df = df.drop_duplicates()
print(df)

tounge = df['tweet'].str.contains(r":P")
df = df.loc[~tounge]

# remove text
df = df.replace(to_replace=[r"\@[\w_]*", # @Username
                            r"http[s]*:\/\/[^\s]*", # URL
                            r":\)|:-\)|: \)|:D|=\)", # smiley face
                            r":\(|:-\(|: \(", #sad face
                            r"[^a-zA-Z# ]+"], # special characters
                value=r"",
                regex=True)

# replace repeated characters with to repetitions
df = df.replace(to_replace=r"([a-zA-Z])(?:\1){2,}",
                value=r"\1\1",
                regex=True)
