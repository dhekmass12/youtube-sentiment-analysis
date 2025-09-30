#you tube comments on  you tube orignal Ai series.
# from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from time import sleep
import undetected_chromedriver as webdriver

import json
import httpx
from textblob import TextBlob
from parsel import Selector
import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

SONG = "sailor_song"
YOUTUBE_TITLE = "Gigi Perez - Sailor Song (Lyrics)"

def sentiment_analysis(review_data):
    # make sure to download the ntlk binaries in the previous snippet
    data = []
    for review in review_data:
        blob = TextBlob(review)
        data.append("Positive" if blob.polarity >= 0.2 else ("Negative" if blob.polarity <= -0.2 else "Neutral"))
    
    return data

driver=webdriver.Chrome()
driver.set_page_load_timeout(10)

driver.get('https://www.youtube.com/')
driver.maximize_window()
sleep(5)
search=driver.find_element(By.NAME,"search_query")
search.clear()
# type name of video or any keyword this will only find the first video of your search 
search.send_keys(YOUTUBE_TITLE)

search.send_keys(Keys.ENTER)
sleep(5)
link=driver.find_element(By.XPATH,"""//*[@id="video-title"]/yt-formatted-string""")
link.click()
sleep(20)
for i in range(300):
    driver.execute_script("window.scrollBy(0,700)","")
    sleep(2)
sleep(20)

comments = []
sentiments = []

comment=driver.find_elements(By.XPATH,"""//*[@id="content-text"]/span""")
print(len(comment))
for i in comment:
    text = i.text
    text = text.split("\n")[0]
    comments.append(text)

sentiments = sentiment_analysis(comments)

df=pd.DataFrame({"comment": comments, "sentiment": sentiments})
df.to_csv("{}.csv".format(SONG),index=False)
assert "No results found." not in driver.page_source
driver.close()