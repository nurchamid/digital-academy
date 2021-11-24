# %%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import nltk
import plotly.figure_factory as ff
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
# from wordcloud import WordCloud
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

st.title('Sentiment Analysis')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase.')

menu = st.sidebar.radio( "What's your go?", ('Explore', 'Predict'))

if menu == 'Explore':
  sentiment = st.sidebar.selectbox("Sentiment ", list_sentiment)

  today = data['tweet_date'].min()
  tomorrow = data['tweet_date'].max()
  start_date = st.sidebar.date_input('Start date', today)
  end_date = st.sidebar.date_input('End date', tomorrow)

  start_date2=pd.to_datetime(start_date)
  end_date2=pd.to_datetime(end_date)

  data=data[(data['tweet_date']>=start_date2)&(data['tweet_date']<=end_date2)]

  # message = st.sidebar.text_area("Input your text", height=20)
  # if message.lower() == '' :
  #   st.sidebar.caption('Sentiment results : none')
  # elif message.lower() in ['smile', 'happy', 'good', 'best'] :
  #   st.sidebar.caption('Sentiment results : positive')
  # elif message.lower() in ['bad'] :
  #   st.sidebar.caption('Sentiment results : negative')
  # else :
  #   st.sidebar.caption('Sentiment results : neutral')

  if sentiment == 'all' :
      # st.write('------------------------------------------------------------')
      total=data.shape[0]
      st.write('Total tweets : ',total)
      st.write('------------------------------------------------------------')
      st.header("Sentiment Trend")

      # Trend
      data['text'] = data['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
      df_trend=data.groupby(['tweet_date', 'sentiment'])['textID'].count().reset_index()
      fig = px.bar(df_trend, x="tweet_date", y="textID", color="sentiment", title="Sentiment Trend", width=800, height=400)
      fig.update_layout(legend=dict(
      orientation="h",
      yanchor="bottom",
      y=1.02,
      xanchor="right",
      x=1),
      yaxis_title=None)
      st.plotly_chart(fig)

      st.write('------------------------------------------------------------')
      st.header("Top Words")

      col1, col2 = st.columns(2)

      df_data_pos = " ".join(data['text'])
      token_text_pos = word_tokenize(df_data_pos)
      bigrams_pos = ngrams(token_text_pos, 1)
      frequency_pos = Counter(bigrams_pos)
      df_pos = pd.DataFrame(frequency_pos.most_common(10))
      df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
      df_pos['total tweet']=df_pos[1]
      df_pos=df_pos.sort_values(by='total tweet', ascending=True)
      figure1 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure1.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col1.plotly_chart(figure1,use_container_width = True)


      # bigram
      df_data_pos = " ".join(data['text'])
      token_text_pos = word_tokenize(df_data_pos)
      bigrams_pos = ngrams(token_text_pos, 2)
      frequency_pos = Counter(bigrams_pos)
      df_pos = pd.DataFrame(frequency_pos.most_common(10))
      df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
      df_pos['total tweet']=df_pos[1]
      df_pos=df_pos.sort_values(by='total tweet', ascending=True)
      figure2 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure2.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col2.plotly_chart(figure2,use_container_width = True)

      st.write('------------------------------------------------------------')
      st.header("Detail Text")

      # fig3 = ff.create_table(data.head(100))
      st.table(data.head(10))

  else : 
      data2=data[data['sentiment']==sentiment]
      # st.write('------------------------------------------------------------')
      total=data2.shape[0]
      st.write('Total tweets : ',total)
      st.write('------------------------------------------------------------')
      st.header("Sentiment Trend")

      #   data2['text'] = data2['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
      #   df_trend=data2.groupby(['tweet_date', 'sentiment'])['textID'].count().reset_index()
      #   st.write(sentiment+" sentiment trend")
      #   fig = px.bar(df_trend, x="tweet_date", y="textID", color="sentiment", width=800, height=400)
      #   fig.update_layout(legend=dict(
      #     orientation="h",
      #     yanchor="bottom",
      #     y=1.02,
      #     xanchor="right",
      #     x=1))
      #   st.plotly_chart(fig)
      
      data2['text'] = data2['text'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
      df_trend=data2.groupby(['tweet_date'])['textID'].count().reset_index().rename(columns={'tweet_date':'Date'})
      fig = px.line(df_trend, x="Date", y="textID")
      fig.update_layout(title_text='Top 10 Bigrams', yaxis_title=None)
      st.plotly_chart(fig)

      st.write('------------------------------------------------------------')
      st.header("Top Words")

      col1, col2 = st.columns(2)

      df_data_pos = " ".join(data2['text'])
      token_text_pos = word_tokenize(df_data_pos)
      bigrams_pos = ngrams(token_text_pos, 1)
      frequency_pos = Counter(bigrams_pos)
      df_pos = pd.DataFrame(frequency_pos.most_common(10))
      df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
      df_pos['total tweet']=df_pos[1]
      df_pos=df_pos.sort_values(by='total tweet', ascending=True)
      figure1 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure1.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col1.plotly_chart(figure1,use_container_width = True)

      # bigram
      df_data_pos = " ".join(data2['text'])
      token_text_pos = word_tokenize(df_data_pos)
      bigrams_pos = ngrams(token_text_pos, 2)
      frequency_pos = Counter(bigrams_pos)
      df_pos = pd.DataFrame(frequency_pos.most_common(10))
      df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
      df_pos['total tweet']=df_pos[1]
      df_pos=df_pos.sort_values(by='total tweet', ascending=True)
      figure2 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure2.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col2.plotly_chart(figure2,use_container_width = True)

      st.write('------------------------------------------------------------')
      st.header("Detail Text")

      st.dataframe(data.head(10)) 

else:
  st.markdown('Predict Data')
  # message = st.sidebar.text_area("Input your text", height=20)
  # if message.lower() == '' :
  #   st.sidebar.caption('Sentiment results : none')
  # elif message.lower() in ['smile', 'happy', 'good', 'best'] :
  #   st.sidebar.caption('Sentiment results : positive')
  # elif message.lower() in ['bad'] :
  #   st.sidebar.caption('Sentiment results : negative')
  # else :
  #   st.sidebar.caption('Sentiment results : neutral')
  with st.form(key='Enter name'):
      message = st.text_input('Enter the name for which you want to know the sentiment')
      # number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)', 0,50,10)
      submit_button = st.form_submit_button(label='Submit')
  if submit_button:
    if message.lower() in ['smile', 'happy', 'good', 'best'] :
      st.markdown('Sentiment results : positive')
    elif message.lower() in ['bad'] :
      st.markdown('Sentiment results : negative')
    else :
      st.markdown('Sentiment results : neutral')
    
    st.markdown('Last 10 Tweet Data')
    st.table(data[data['text'].str.contains(message)].sort_values(by='tweet_date', ascending=False).head(10))
