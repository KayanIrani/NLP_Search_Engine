import { useState } from 'react';
import { Search, Newspaper, TrendingUp, Database, Activity } from 'lucide-react';

function App() {
  const [query, setQuery] = useState('');
  const [numResults, setNumResults] = useState(3);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [stats, setStats] = useState(null);
  const [bertQuery, setBertQuery] = useState('');
  const [bertNumResults, setBertNumResults] = useState(3);
  const [bertResults, setBertResults] = useState(null);
  const [bertLoading, setBertLoading] = useState(false);
  const [bertError, setBertError] = useState('');

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          n: numResults,
        }),
      });

      if (!res.ok) {
        throw new Error('Search failed');
      }

      const data = await res.json();
      setResults(data);
    } catch (err) {
      setError('Failed to search. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleBertSearch = async () => {
    if (!bertQuery.trim()) {
      setBertError('Please enter a search query');
      return;
    }

    setBertLoading(true);
    setBertError('');
    setBertResults(null);

    try {
      const res = await fetch('/api/search/bert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: bertQuery,
          n: bertNumResults,
        }),
      });

      if (!res.ok) {
        throw new Error('BERT search failed');
      }

      const data = await res.json();
      setBertResults(data);
    } catch (err) {
      setBertError('Failed to search. Please try again.');
      console.error(err);
    } finally {
      setBertLoading(false);
    }
  };

  const handleBertKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleBertSearch();
    }
  };

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <div className="bg-zinc-900 border-b border-zinc-800">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-zinc-800 p-3 rounded-xl border border-zinc-700">
                <Newspaper className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">News Search Engine</h1>
                <p className="text-zinc-400 text-sm">Powered by BM25 Algorithm</p>
              </div>
            </div>
            <button
              onClick={fetchStats}
              className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg transition-all duration-200 border border-zinc-700"
            >
              <Database className="w-4 h-4" />
              Stats
            </button>
          </div>
        </div>
      </div>

      {/* Stats Modal */}
      {stats && (
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Corpus Statistics
              </h3>
              <button
                onClick={() => setStats(null)}
                className="text-zinc-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700">
                <p className="text-zinc-400 text-sm">Total Documents</p>
                <p className="text-2xl font-bold text-white">{stats.total_documents}</p>
              </div>
              <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700">
                <p className="text-zinc-400 text-sm">Corpus Size</p>
                <p className="text-2xl font-bold text-white">{stats.corpus_size}</p>
              </div>
              <div className="bg-zinc-800 rounded-lg p-4 border border-zinc-700">
                <p className="text-zinc-400 text-sm">Categories</p>
                <p className="text-2xl font-bold text-white">
                  {stats.categories ? Object.keys(stats.categories).length : 0}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        {/* Search Section */}
        <div className="bg-zinc-900 rounded-2xl p-8 border border-zinc-800">
          <div className="mb-6">
            <label className="block text-zinc-400 text-sm font-medium mb-2">
              Search Query
            </label>
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter your search query..."
                className="w-full px-4 py-4 pl-12 bg-zinc-800 border border-zinc-700 rounded-xl text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent transition-all"
              />
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-zinc-500" />
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-zinc-400 text-sm font-medium mb-2">
              Number of Results
            </label>
            <input
              type="number"
              value={numResults}
              onChange={(e) => setNumResults(Math.max(1, parseInt(e.target.value) || 1))}
              min="1"
              max="20"
              className="w-full px-4 py-4 bg-zinc-800 border border-zinc-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-zinc-600 focus:border-transparent transition-all"
            />
          </div>

          {error && (
            <div className="mb-6 bg-red-950 border border-red-900 rounded-lg p-4">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          <button
            onClick={handleSearch}
            disabled={loading}
            className="w-full bg-white hover:bg-zinc-100 text-black font-semibold py-4 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-zinc-400 border-t-black rounded-full animate-spin" />
                Searching...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Search News
              </>
            )}
          </button>
        </div>

        {/* Results Section */}
        {results && results.articles && results.articles.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="w-5 h-5 text-zinc-400" />
              <h2 className="text-2xl font-bold text-white">
                Search Results ({results.articles.length})
              </h2>
            </div>

            <div className="space-y-4">
              {results.articles.map((article, index) => (
                <div
                  key={index}
                  className="bg-zinc-900 rounded-xl p-6 border border-zinc-800 hover:border-zinc-700 transition-all duration-200 hover:scale-[1.01]"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="bg-white text-black w-10 h-10 rounded-lg flex items-center justify-center font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">Article {index + 1}</h3>
                        <p className="text-sm text-zinc-400">
                          Relevance Score: {results.scores[index].toFixed(4)}
                        </p>
                      </div>
                    </div>
                  </div>
                  <p className="text-zinc-300 leading-relaxed">
                    {article.length > 500 ? `${article.substring(0, 500)}...` : article}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {results && results.articles && results.articles.length === 0 && (
          <div className="mt-8 bg-zinc-900 rounded-xl p-12 border border-zinc-800 text-center">
            <div className="bg-zinc-800 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-zinc-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No Results Found</h3>
            <p className="text-zinc-400">Try adjusting your search query</p>
          </div>
        )}
      </div>

      {/* BERT + BM25 Search Section */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="bg-gradient-to-br from-emerald-950 to-green-950 rounded-2xl p-8 border-2 border-emerald-800">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-emerald-700 p-3 rounded-xl">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">BERT + BM25 Hybrid Search</h2>
              <p className="text-emerald-300 text-sm">Advanced semantic search with BM25 reranking</p>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-emerald-300 text-sm font-medium mb-2">
              Search Query
            </label>
            <div className="relative">
              <input
                type="text"
                value={bertQuery}
                onChange={(e) => setBertQuery(e.target.value)}
                onKeyPress={handleBertKeyPress}
                placeholder="Enter your search query for BERT analysis..."
                className="w-full px-4 py-4 pl-12 bg-emerald-950 border border-emerald-700 rounded-xl text-white placeholder-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-transparent transition-all"
              />
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-emerald-500" />
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-emerald-300 text-sm font-medium mb-2">
              Number of Results
            </label>
            <input
              type="number"
              value={bertNumResults}
              onChange={(e) => setBertNumResults(Math.max(1, parseInt(e.target.value) || 1))}
              min="1"
              max="20"
              className="w-full px-4 py-4 bg-emerald-950 border border-emerald-700 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-transparent transition-all"
            />
          </div>

          {bertError && (
            <div className="mb-6 bg-red-950 border border-red-900 rounded-lg p-4">
              <p className="text-red-400 text-sm">{bertError}</p>
            </div>
          )}

          <button
            onClick={handleBertSearch}
            disabled={bertLoading}
            className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {bertLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-emerald-300 border-t-white rounded-full animate-spin" />
                Searching with BERT...
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                Search with BERT
              </>
            )}
          </button>
        </div>

        {/* BERT Results Section */}
        {bertResults && bertResults.articles && bertResults.articles.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              <h2 className="text-2xl font-bold text-white">
                BERT Results ({bertResults.articles.length})
              </h2>
            </div>

            <div className="space-y-4">
              {bertResults.articles.map((article, index) => (
                <div
                  key={index}
                  className="bg-gradient-to-br from-emerald-950 to-green-950 rounded-xl p-6 border-2 border-emerald-800 hover:border-emerald-700 transition-all duration-200 hover:scale-[1.01]"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="bg-emerald-600 text-white w-10 h-10 rounded-lg flex items-center justify-center font-bold">
                        {index + 1}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-white">Article {index + 1}</h3>
                        <p className="text-sm text-emerald-300">
                          Relevance Score: {bertResults.scores[index].toFixed(4)}
                        </p>
                      </div>
                    </div>
                  </div>
                  <p className="text-emerald-50 leading-relaxed">
                    {article.length > 500 ? `${article.substring(0, 500)}...` : article}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {bertResults && bertResults.articles && bertResults.articles.length === 0 && (
          <div className="mt-8 bg-gradient-to-br from-emerald-950 to-green-950 rounded-xl p-12 border-2 border-emerald-800 text-center">
            <div className="bg-emerald-800 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-emerald-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">No Results Found</h3>
            <p className="text-emerald-300">Try adjusting your search query</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;