from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
import os
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import PorterStemmer
from typing import List

# Initialize FastAPI app
app = FastAPI(title="News Search API", description="BM25-based news search engine")

# Global variables for corpus and data
corpus = []
data = None
bm25 = None

# Pydantic models
class SearchQuery(BaseModel):
    query: str
    n: int = 3

class SearchResult(BaseModel):
    articles: List[str]
    scores: List[float]

# Preprocessing functions
def spl_chars_removal(lst):
    lst1 = []
    for element in lst:
        str_cleaned = re.sub("[^0-9a-zA-Z]", " ", element)
        lst1.append(str_cleaned)
    return lst1

def stopwords_removal(lst):
    lst1 = []
    for text in lst:
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if word not in STOPWORDS]
        str_t = " ".join(tokens_without_sw)
        lst1.append(str_t)
    return lst1

def preprocess_text(text):
    """Preprocess a single text document"""
    ps = PorterStemmer()
    
    # Tokenization
    words = word_tokenize(text)
    
    # Special characters removal
    words = spl_chars_removal(words)
    
    # Stopwords removal
    words = stopwords_removal(words)
    
    # Stemming
    final_lst = []
    for word in words:
        final_lst.append(ps.stem(word))
    
    return ' '.join(final_lst)

@app.on_event("startup")
async def startup_event():
    """Initialize NLTK downloads and load data on startup"""
    global corpus, data, bm25
    
    # Download NLTK data
    curr_dir = os.getcwd()
    down_dir = os.path.join(curr_dir, 'nltk_data')
    
    try:
        nltk.download('punkt', download_dir=down_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=down_dir, quiet=True)
    except:
        pass
    
    # Load data
    try:
        data = pd.read_csv(os.path.join('data', 'BBC News Train.csv'))
        
        # Build corpus
        print("Building corpus...")
        corpus = []
        for news in data['Text']:
            corpus.append(preprocess_text(news))
        
        # Initialize BM25
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"Corpus built with {len(corpus)} documents")
    except FileNotFoundError:
        print("Warning: BBC News Train.csv not found. Search endpoint will not work.")
    except Exception as e:
        print(f"Error loading data: {e}")

@app.get("/api")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "News Search API",
        "endpoints": {
            "/api/search": "POST - Search for news articles",
            "/api/health": "GET - Check API health",
            "/api/stats": "GET - Get corpus statistics"
        }
    }

@app.get("/api/health")
async def health_check():
    """Check if the API is ready"""
    if data is None or bm25 is None:
        raise HTTPException(status_code=503, detail="Service not ready - data not loaded")
    return {
        "status": "healthy",
        "corpus_size": len(corpus)
    }

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the corpus"""
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_documents": len(data),
        "corpus_size": len(corpus),
        "categories": data['Category'].value_counts().to_dict() if 'Category' in data.columns else {}
    }

@app.post("/api/search", response_model=SearchResult)
async def search_news(search_query: SearchQuery):
    """
    Search for news articles using BM25 algorithm
    
    - **query**: Search query string
    - **n**: Number of top results to return (default: 3)
    """
    if data is None or bm25 is None:
        raise HTTPException(status_code=503, detail="Service not ready - data not loaded")
    
    if not search_query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if search_query.n < 1:
        raise HTTPException(status_code=400, detail="n must be at least 1")
    
    # Preprocess query
    processed_query = preprocess_text(search_query.query.lower())
    tokenized_query = processed_query.split(" ")
    
    # Get BM25 scores
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get top n results
    n = min(search_query.n, len(corpus))  # Don't exceed corpus size
    ind = np.argpartition(doc_scores, -n)[-n:]
    
    # Sort by score (descending)
    ind = ind[np.argsort(-doc_scores[ind])]
    
    # Get articles and scores
    articles = [data['Text'].iloc[i] for i in ind]
    scores = [float(doc_scores[i]) for i in ind]
    
    return SearchResult(articles=articles, scores=scores)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)