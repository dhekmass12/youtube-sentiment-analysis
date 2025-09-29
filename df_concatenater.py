import pandas as pd
import nltk
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('die_with_a_smile.csv')
df2 = pd.read_csv('in_the_stars.csv')
df3 = pd.read_csv('rewrite_the_stars.csv')
df4 = pd.read_csv('someone_like_you.csv')
df5 = pd.read_csv('talking_to_the_moon.csv')
df6 = pd.read_csv('thats_what_i_like.csv')
df7 = pd.read_csv('the_lazy_song.csv')
df8 = pd.read_csv('we_dont_talk_anymore.csv')
df9 = pd.read_csv('youtube_rewind_2017.csv')
df10 = pd.read_csv('youtube_rewind_2018.csv')
df11 = pd.read_csv('youtube_rewind_2019.csv')


df = pd.concat([
    df, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11
])

df.to_csv("{}.csv".format("sentiment_analysis"),index=False)