# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:35:59 2021

@author: Harsh Garg
"""
#reference article https://medium.com/neuronio/from-sentiment-analysis-to-emotion-recognition-a-nlp-story-bcc9d6ff61ae
#docs https://docs.tweepy.org/en/stable/getting_started.html

import pandas as pd
import tweepy
# import emoji
import re

#Twitter API credentials

consumer_key = '8j2lF75V9cxwirWaD80QaV6dS'
consumer_secret = 'IHQxiL2PtOabYXoBHx6fhBCWK0irGrxK0jiy0GqWPvEyzgf6K5'
access_token = '1454139798080274436-LG62e7BYFAKdoWxtMMwHyVa9t9wGEA'
access_token_secret = 'ocsdPe1h701ztX82RZoKQsQmWru4PbshUJtC98KYi1Q00'

auth = tweepy.OAuthHandler(consumer_key = consumer_key, consumer_secret = consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#helper functions to clean and function to scrape text respectively

def cjk_detect(texts): #cjk - chinese, japanese, korean
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    return ''

def lang_detect_and_drop(text):
    langs = []
    final_text = []
    for word in text:
        if (word.startswith('@') or word.startswith('http') or word.startswith('www.')):
            pass
        elif word.startswith('#'):
            langs.append(cjk_detect(word[1:]))
            final_text.append(word[1:])
        else:
            langs.append(cjk_detect(word))
            final_text.append(word)
    return langs, final_text

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def remove_spam_hashtags(df):
    df['hashtags_start'] = ''
    df['hashtags_end'] = ''
    for ind in range(len(df)):
        inds = []
        hashes = dict(df['entities'].loc[ind])['hashtags']
        number_of_hashes = len(hashes)
        if number_of_hashes == 0: continue
        for x in range(number_of_hashes):
            inds.append(hashes[x]['indices'][0])
            inds.append(hashes[x]['indices'][1])
        inds_size = len(inds)
        if inds_size==2:
            #do nothing
            continue
        else:
            longest_chain_first = inds[0]
            longest_chain_last = inds[1]
            longest_chain_alt_first, longest_chain_alt_last = 0,0
            for pointer in range(2,inds_size-1):
                #check if its the first element of the start-end pair (even indexed)
                if pointer%2==0:
                    #check if the hashtags are continuous or not
                    if inds[pointer] - inds[pointer-1]==1:
                        #check which vars were last used
                        if longest_chain_last>longest_chain_alt_last:
                            longest_chain_last = inds[pointer+1]
                        else:
                            longest_chain_alt_last = inds[pointer+1]
                    else:
                        #Case 1: Encounter hashtags breaking sequence for the first time
                        if longest_chain_alt_first - longest_chain_alt_last == 0: 
                            longest_chain_alt_first, longest_chain_alt_last = inds[pointer], inds[pointer + 1]
                        #Case 2: Generalised case 1 part 1: alternate chain longer than main chain -> replace the main chain
                        #with alternate chain, and then re-initialise alternate chain
                        else:
                            if longest_chain_last - longest_chain_first < longest_chain_alt_last - longest_chain_alt_first:
                                longest_chain_first = longest_chain_alt_first
                                longest_chain_last = longest_chain_alt_last
                                longest_chain_alt_first, longest_chain_alt_last = inds[pointer], inds[pointer + 1]
                            elif longest_chain_last - longest_chain_first >= longest_chain_alt_last - longest_chain_alt_first:
                                longest_chain_alt_first, longest_chain_alt_last = inds[pointer], inds[pointer + 1]             
                else: pass
        if longest_chain_last > longest_chain_alt_last:
            try:
                df['RT_full_length'].loc[ind] = df['RT_full_length'].loc[ind][:longest_chain_first]+df['RT_full_length'].loc[ind][longest_chain_last+1:]
            except:
                df['RT_full_length'].loc[ind] = df['RT_full_length'].loc[ind][:longest_chain_first]+df['RT_full_length'].loc[ind][longest_chain_last:]
        else:
            try:
                df['RT_full_length'].loc[ind] = df['RT_full_length'].loc[ind][:longest_chain_alt_first]+df['RT_full_length'].loc[ind][longest_chain_alt_last+1:]
            except:
                df['RT_full_length'].loc[ind] = df['RT_full_length'].loc[ind][:longest_chain_alt_first]+df['RT_full_length'].loc[ind][longest_chain_alt_last:]
    return df
                
def keyword_dataframe_append(keyword, api, df = pd.DataFrame()):
    word = api.search_tweets("#"+keyword, lang='en', count = 100, tweet_mode="extended")
    tweets = []
    dict_keys = ['created_at', 'id', 'id_str', 'full_text', 'truncated', 'entities', 
                 'extended_entities', 'metadata', 'source', 'in_reply_to_status_id', 
                 'in_reply_to_status_id_str', 'in_reply_to_user_id', 
                 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 
                 'coordinates', 'place', 'contributors', 'retweeted_status', 
                 'is_quote_status', 'retweet_count', 'favorite_count', 'favorited', 
                 'retweeted', 'possibly_sensitive', 'lang']
    for tweet in word:
        tweets.append([tweet._json.get(key) for key in dict_keys])
    
    tweets = pd.DataFrame(tweets, columns = dict_keys)
    tweets['Keyword'] = keyword
    tweets['RT_full_length'] = ''

    for ind in range(len(tweets)):
        try:
            tweets['RT_full_length'].loc[ind] = tweets['retweeted_status'].loc[ind]['full_text']
        except:
            tweets['RT_full_length'].loc[ind] = tweets['full_text'].loc[ind]
    
    tweets = remove_spam_hashtags(tweets)
    tweets['len_full_text'] = tweets['RT_full_length'].apply(lambda x: len(x))
    tweets['deEmojify'] = tweets['RT_full_length'].apply(lambda x: deEmojify(x))
    tweets['deEmojify'] = tweets['deEmojify'].apply(lambda x: x.split())
    
    tweets['final_result'] = ''
    tweets['final_langs'] = ''
    
    for ind in range(len(tweets['deEmojify'])):
        lang, fin_text = lang_detect_and_drop(tweets['deEmojify'].loc[ind])
        tweets['final_result'].loc[ind] = fin_text
        tweets['final_langs'].loc[ind] = lang

    return pd.concat([df, tweets])

def dataframe_clean(df):
    df = df.drop_duplicates(subset = 'RT_full_length').reset_index(drop = True)
    df['cleaned_text'] = ''
    df['cleaned_text'] = df['final_result'].apply(lambda x: ' '.join(x))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.replace('&amp;','&'))
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.replace('$RT',''))
    emotion = {'joy':'joy',
               'excited':'joy',
               'happy':'joy',
               'fear':'fear',
               'scared':'fear',
               'afraid':'fear',
               'anticipation':'anticipation',
               'hope':'anticipation',
               'trust':'trust',
               'pride':'pride',
               ''}
    df['emotion'] = df['Keyword'].apply(lambda x: emotion.get(x))
    return df

#Generating scraped dataset using keywords
df = keyword_dataframe_append('joy', api)
df = keyword_dataframe_append('excited', api, df)
df = keyword_dataframe_append('happy', api, df)
df = keyword_dataframe_append('fear', api, df)
df = keyword_dataframe_append('scared', api, df)
df = keyword_dataframe_append('afraid', api, df)
df = keyword_dataframe_append('anticipation', api, df)
df = keyword_dataframe_append('hope', api, df)
df = keyword_dataframe_append('trust', api, df)
df = keyword_dataframe_append('pride', api, df)
df = dataframe_clean(df)

df.to_excel(r'/Users/harshgarg/Desktop/intract/Text Analysis/TwitterAPI/data.xlsx',index=False)
