import nltk
from nltk.tokenize import word_tokenize
import os
import re
from gensim.parsing.preprocessing import STOPWORDS

curr_dir = os.getcwd()
down_dir = os.path.join(curr_dir,'nltk_data')

nltk.download('punkt',download_dir=down_dir)
nltk.download('punkt_tab',download_dir=down_dir)

sentence = "Data Science and Machine Learning are powerful technologies"
words = word_tokenize(sentence)
print(words)

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
sentence = ["This is an interesting fact about the dinosaurs"]
print(''.join(stopwords_removal(sentence))
)


