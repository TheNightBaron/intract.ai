from transformers import pipeline
import pandas as pd
import numpy as np

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def create_dataset():
    #Import data
    emails = pd.read_excel(r"C:\Users\Harsh Garg\Desktop\intract\Email Subject Line Scrapper.xlsx",'Saved Data')

    #Delete irrelevant columns and add relevant columns
    emails = emails[['Subject Line']]
#     emails = emails.drop_duplicates(subset = 'Subject Line')
    
    return list(emails['Subject Line'])

data = create_dataset()

labels = ['fear','anticipation','joy','trust','pride']

emails = pd.read_excel(r"C:\Users\Harsh Garg\Desktop\intract\Email Subject Line Scrapper.xlsx",'Saved Data')
emails = emails[['Subject Line']]

for label in labels:
    emails[label] = ''

emails['prediction'] = ''

for row in range(len(emails)):
    prediction = classifier(data[row], labels, multi_label=True, num_workers = 4)
    prediction['probab'] = softmax(prediction.get('scores'))
    for x in zip(prediction['labels'], prediction['probab']):
        emails[x[0]].loc[row] = x[1]

dictionary = {0:'fear',
              1:'anticipation',
              2:'joy',
              3:'trust',
              4:'pride'}

for x in range(len(emails)):
    probs = list(emails[['fear','anticipation','joy','trust','pride']].loc[x])
    emails['prediction'].loc[x] = dictionary.get(np.where(probs==max(probs))[0][0])

emails.to_excel(r"C:\Users\Harsh Garg\Desktop\intract\HuggingFace_pre_trained_bart_large_mnli.xlsx",
                index=False)

