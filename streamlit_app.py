%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import nltk
import plotly.figure_factory as ff
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

st.set_page_config(layout = "wide")

# get data 
data=pd.read_csv('https://raw.githubusercontent.com/nurchamid/dashboard_streamlit/main/train_sentiment_extraction.csv')

# data preparation
def clean_text(tweets):
    
    # Replacing @handle with the word USER
    tweets_handle = tweets.str.replace(r'@[\S]+', 'user')
    
    # Replacing the Hast tag with the word hASH
    tweets_hash = tweets_handle.str.replace(r'#(\S+)','hash')
    
    # Removing the all the Retweets
    tweets_r = tweets_hash.str.replace(r'\brt\b',' ')
    
    # Replacing the URL or Web Address
    tweets_url = tweets_r.str.replace(r'((www\.[\S]+)|(http?://[\S]+))','URL')
    
    # Replacing Two or more dots with one
    tweets_dot = tweets_url.str.replace(r'\.{2,}', ' ')
    
    # Removing all the special Characters
    tweets_special = tweets_dot.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    tweets_ascii = tweets_special.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    tweets_space = tweets_ascii.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    Dataframe = tweets_space.str.replace(r'\s+',' ')
    
    return Dataframe

data['text'] = clean_text(data['text'])
data['text'] = data['text'].apply(str)
data['text'] = data['text'].str.lower()
data['selected_text'] = clean_text(data['selected_text'])
data['selected_text'] = data['selected_text'].str.lower()

start_date=pd.to_datetime('2021-08-01')
end_date=pd.to_datetime('2021-08-08')
data["tweet_date"] = np.random.choice(pd.date_range(start_date, end_date), len(data))

list_sentiment=['all']+list(data['sentiment'].unique())

# dashboard preparation

st.header("Sentiment Analysis")

sentiment = st.sidebar.selectbox("Sentiment ", list_sentiment)

today = data['tweet_date'].min()
tomorrow = data['tweet_date'].max()
start_date = st.sidebar.date_input('Start date', today)
end_date = st.sidebar.date_input('End date', tomorrow)

start_date2=pd.to_datetime(start_date)
end_date2=pd.to_datetime(end_date)

data=data[(data['tweet_date']>=start_date2)&(data['tweet_date']<=end_date2)]

message = st.sidebar.text_area("Input your text", height=20)
if message.lower() == '' :
  st.sidebar.caption('Sentiment results : none')
elif message.lower() in ['smile', 'happy', 'good', 'best'] :
  st.sidebar.caption('Sentiment results : positive')
elif message.lower() in ['bad'] :
  st.sidebar.caption('Sentiment results : negative')
else :
  st.sidebar.caption('Sentiment results : neutral')

if sentiment == 'all' :
  st.write('------------------------------------------------------------')
  total=data.shape[0]
  st.write('Total tweets : ',total)
  
  # Trend
  data['text'] = data['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
  df_trend=data.groupby(['tweet_date', 'sentiment'])['textID'].count().reset_index()
  fig = px.bar(df_trend, x="tweet_date", y="textID", color="sentiment", title="Sentiment Trend", width=800, height=400)
  fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
  st.plotly_chart(fig)

  col1, col2 = st.columns(2)

  # Word cloud of the text with the negative sentiment
  k = (' '.join(data['text']))
  wordcloud = WordCloud(width = 800, height = 600, background_color = 'white').generate(k)
  col1.plotly_chart(px.imshow(wordcloud, title='Top Word'))

  # bigram
  df_data_pos = " ".join(data['text'])
  token_text_pos = word_tokenize(df_data_pos)
  bigrams_pos = ngrams(token_text_pos, 2)
  frequency_pos = Counter(bigrams_pos)
  df_pos = pd.DataFrame(frequency_pos.most_common(10))
  df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
  df_pos['count']=df_pos[1]
  fig = px.bar(df_pos, x='count', y='word', orientation='h')
  fig.update_layout(title_text='Top 10 Bigrams',yaxis={'categoryorder':'total ascending'}, width=1000, height=450)
  col2.plotly_chart(fig,use_container_width = True)

  # fig = ff.create_table(data.head())
  # fig.show()  
  st.dataframe(data.head(2))  

else : 
  data2=data[data['sentiment']==sentiment]
  st.write('------------------------------------------------------------')
  total=data2.shape[0]
  st.write('Total tweets : ',total)

  data2['text'] = data2['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
  df_trend=data2.groupby(['tweet_date', 'sentiment'])['textID'].count().reset_index()
  st.write(sentiment+" sentiment trend")
  fig = px.bar(df_trend, x="tweet_date", y="textID", color="sentiment", width=800, height=400)
  fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
  st.plotly_chart(fig)

  col1, col2 = st.columns(2)

  # Word cloud of the text with the X sentiment
  k = (' '.join(data2['text']))
  wordcloud = WordCloud(width = 800, height = 600, background_color = 'white').generate(k)
  col1.plotly_chart(px.imshow(wordcloud, title='Top Word'))

  # bigram
  df_data_pos = " ".join(data2['text'])
  token_text_pos = word_tokenize(df_data_pos)
  bigrams_pos = ngrams(token_text_pos, 2)
  frequency_pos = Counter(bigrams_pos)
  df_pos = pd.DataFrame(frequency_pos.most_common(10))
  df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
  df_pos['count']=df_pos[1]
  fig = px.bar(df_pos, x='count', y='word', orientation='h')
  fig.update_layout(title_text='Top 10 Bigrams',yaxis={'categoryorder':'total ascending'}, width=1000, height=450)
  col2.plotly_chart(fig,use_container_width = True)

  st.dataframe(data.head(2)) 
