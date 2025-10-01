import pandas as pd
import nltk
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from langdetect import detect
import os


dfs = []

directory_path = 'dataframes'  # Current directory, or specify a path like 'my_folder'
all_entries = os.listdir(directory_path)
files = [entry for entry in all_entries if os.path.isfile(os.path.join(directory_path, entry))]

for file in files:
    df = pd.read_csv("dataframes/" + file)
    dfs.append(df)

df = pd.concat(dfs)

# # lower all chars
df["comment"] = df["comment"].str.lower()

# retain only a-z and space
df["comment"] = df['comment'].apply(lambda x: re.sub(r'[^a-z ]', '', str(x)))

# remove single char in the beginning, middle, and end
df["comment"] = df['comment'].apply(lambda x: re.sub(r'(^|\s)[a-z]\b', '', str(x)))

# remove leading and trailing spaces
df["comment"] = df['comment'].apply(lambda x: x.strip())

# replace multiple spaces with a single space
df["comment"] = df['comment'].apply(lambda x:  re.sub(r'\s\s+', '', str(x)))

# remove missing values
df = df[(df['comment'] != "") & (df['comment'].notnull()) & (df['comment'] != "nan")]

# remove non-english comments
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

df = df[df['comment'].apply(is_english)]

df.to_csv("{}.csv".format("sentiment_analysis"),index=False)