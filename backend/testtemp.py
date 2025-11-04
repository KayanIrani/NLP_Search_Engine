from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(title="News Search API", description="BM25 and BERT-based news search engine")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for corpus and data
corpus = []
data = None
bm25 = None
bert_model = None
original_embeddings = None

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
    global corpus, data, bm25, bert_model, original_embeddings
    
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
        
        # Build corpus for BM25
        print("Building BM25 corpus...")
        corpus = []
        for news in data['Text']:
            corpus.append(preprocess_text(news))
        
        # Initialize BM25
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"BM25 corpus built with {len(corpus)} documents")
        
        # Load BERT model and create embeddings
        print("Loading BERT model...")
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Creating document embeddings...")
        original_embeddings = bert_model.encode(data['Text'].tolist(), show_progress_bar=True)
        
        print(f"BERT embeddings created for {len(original_embeddings)} documents")
        
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
            "/api/search": "POST - Search for news articles using BM25",
            "/api/search/bert": "POST - Search using BERT + BM25 hybrid",
            "/api/health": "GET - Check API health",
            "/api/stats": "GET - Get corpus statistics"
        }
    }

@app.get("/api/health")
async def health_check():
    """Check if the API is ready"""
    if data is None or bm25 is None:
        raise HTTPException(status_code=503, detail="Service not ready - data not loaded")
    
    bert_status = "loaded" if bert_model is not None else "not loaded"
    
    return {
        "status": "healthy",
        "corpus_size": len(corpus),
        "bert_model": bert_status
    }

@app.get("/api/stats")
async def get_stats():
    """Get statistics about the corpus"""
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_documents": len(data),
        "corpus_size": len(corpus),
        "categories": data['Category'].value_counts().to_dict() if 'Category' in data.columns else {},
        "bert_enabled": bert_model is not None
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
    n = min(search_query.n, len(corpus))
    ind = np.argpartition(doc_scores, -n)[-n:]
    
    # Sort by score (descending)
    ind = ind[np.argsort(-doc_scores[ind])]
    
    # Get articles and scores
    articles = [data['Text'].iloc[i] for i in ind]
    scores = [float(doc_scores[i]) for i in ind]
    
    return SearchResult(articles=articles, scores=scores)

@app.post("/api/search/bert", response_model=SearchResult)
async def search_news_bert(search_query: SearchQuery):
    """
    Search for news articles using BM25 filtering + BERT semantic reranking
    
    This is more efficient: BM25 quickly filters to top candidates,
    then BERT does semantic understanding on smaller set.
    
    Example: Query "car" will:
    1. BM25 finds docs with "car", "cars", "vehicle" (fast keyword match)
    2. BERT reranks to also surface "automobile", "sedan" (semantic similarity)
    
    - **query**: Search query string
    - **n**: Number of top results to return (default: 3)
    """
    if data is None or bert_model is None or original_embeddings is None:
        raise HTTPException(status_code=503, detail="BERT service not ready")
    
    if not search_query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if search_query.n < 1:
        raise HTTPException(status_code=400, detail="n must be at least 1")
    
    # Step 1: BM25 fast filtering - get top 100 candidates
    # This reduces load on BERT significantly
    processed_query = preprocess_text(search_query.query.lower())
    tokenized_query = processed_query.split(" ")
    
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top 100 candidates from BM25 (or less if corpus is smaller)
    top_k_bm25 = min(100, len(corpus))
    bm25_top_indices = np.argpartition(bm25_scores, -top_k_bm25)[-top_k_bm25:]
    
    print(f"BM25 filtered to {len(bm25_top_indices)} candidates")
    
    # Step 2: BERT semantic reranking on BM25 candidates only
    # Now BERT only processes 100 docs instead of all docs
    query_embedding = bert_model.encode([search_query.query])
    
    # Get embeddings only for BM25 candidates
    candidate_embeddings = original_embeddings[bm25_top_indices]
    
    # Calculate semantic similarity
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
    
    print(f"BERT reranked {len(similarities)} candidates")
    
    # Step 3: Use BERT scores directly (since BM25 already filtered)
    # BERT captures semantic meaning like "car" → "automobile"
    
    # Get top n results based on BERT semantic similarity
    n = min(search_query.n, len(similarities))
    top_n_indices = np.argsort(-similarities)[:n]
    
    # Map back to original indices
    final_indices = bm25_top_indices[top_n_indices]
    
    # Get articles and scores
    articles = [data['Text'].iloc[i] for i in final_indices]
    scores = [float(similarities[i]) for i in top_n_indices]
    
    return SearchResult(articles=articles, scores=scores)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)