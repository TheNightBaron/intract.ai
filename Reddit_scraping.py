# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:20:34 2021

@author: Harsh Garg
"""

#Reference_article
#https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
#https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
#https://towardsdatascience.com/a-guide-to-text-classification-and-sentiment-analysis-2ab021796317
#https://towardsdatascience.com/automate-sentiment-analysis-process-for-reddit-post-textblob-and-vader-8a79c269522f

import pandas as pd
import praw
import json

client_id = 'STB-pOcnpygdNba9hyfs0A'
client_secret = 'UCzcHav7Aygr_Lnni4hru8vOgXvMLQ'
user_agent = 'Test App' 
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

posts = []
ml_subreddit = reddit.subreddit('funny')
for post in ml_subreddit.hot(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
