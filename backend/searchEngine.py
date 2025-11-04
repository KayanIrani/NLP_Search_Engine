import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from rank_bm25 import *
import warnings
import re
import os
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer

curr_dir = os.getcwd()
down_dir = os.path.join(curr_dir,'nltk_data')

nltk.download('punkt',download_dir=down_dir)
nltk.download('punkt_tab',download_dir=down_dir)


data = pd.read_csv(os.path.join('data','BBC News Train.csv'))

def spl_chars_removal(lst):
    lst1=list()
    for element in lst:
        str=""
        str = re.sub("[^0-9a-zA-Z]"," ",element)
        lst1.append(str)
    return lst1

def stopwords_removal(lst):
    lst1=list()
    for str in lst:
        text_tokens = word_tokenize(str)
        tokens_without_sw = [word for word in text_tokens if not word in STOPWORDS]
        str_t = " ".join(tokens_without_sw)
        lst1.append(str_t)
 
    return lst1

ps = PorterStemmer() 

corpus = []
for news in data['Text']:
  # Tokenization
  words = word_tokenize(news)

  # spl chars removal
  words = spl_chars_removal(words)

  # stopwords removal
  words = stopwords_removal(words)

  # stemming
  final_lst = []
  for word in words:
      final_lst.append(ps.stem(word))
        
  corpus.append(' '.join(final_lst))

def news_search(query, corpus, n, news_data):
  tokenized_corpus = [doc.split(" ") for doc in corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  tokenized_query = query.split(" ")
  doc_scores = bm25.get_scores(tokenized_query)
  print(doc_scores)
  ind = np.argpartition(doc_scores, -n)[-n:]
  return news_data['Text'][ind]


Query = input("Enter your query for news search: ").lower()
n = 3
print(f'Displaying top {n} articles \n')
result = news_search(Query, corpus, n, data)
for ans in result:
  print(ans)
  print()
