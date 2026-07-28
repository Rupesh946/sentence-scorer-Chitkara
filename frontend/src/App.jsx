import React, { useState } from 'react';
import axios from 'axios';
import { Target, CheckCircle2, AlertCircle, Loader2, Sparkles } from 'lucide-react';

function App() {
  const [sentence, setSentence] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleScore = async () => {
    if (!sentence.trim()) return;
    
    setLoading(true);
    setError(null);
    try {
      const endpoint = import.meta.env.PROD ? '/api/evaluate' : 'http://localhost:8000/api/evaluate';
      const response = await axios.post(endpoint, {
        sentence: sentence
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred while connecting to the server.");
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score, max) => {
    const ratio = score / max;
    if (ratio >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (ratio >= 0.5) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-red-600 bg-red-50 border-red-200';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-6 font-sans text-gray-800">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-4 pt-12">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-100 rounded-full mb-4 shadow-inner">
            <Target className="w-10 h-10 text-indigo-600" />
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">
            Objective Sentence Scorer
          </h1>
          <p className="text-lg text-gray-500 max-w-2xl mx-auto font-medium">
            Evaluate your learning objectives based on Bloom's Taxonomy and the ABCD model.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-2xl shadow-xl shadow-indigo-100/50 p-8 border border-gray-100 transition-all hover:shadow-2xl hover:shadow-indigo-100/60 duration-300">
          <div className="space-y-4">
            <label htmlFor="sentence" className="block text-sm font-semibold text-gray-700 uppercase tracking-wider">
              Learning Objective
            </label>
            <textarea
              id="sentence"
              rows="3"
              className="w-full rounded-xl border-gray-200 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 p-4 text-lg bg-gray-50/50 placeholder:text-gray-400 transition-colors"
              placeholder='e.g., "Students will be able to analyze data using Excel with 90% accuracy."'
              value={sentence}
              onChange={(e) => setSentence(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleScore();
                }
              }}
            />
            <div className="flex justify-end">
              <button
                onClick={handleScore}
                disabled={loading || !sentence.trim()}
                className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-xl shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform active:scale-95"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" />
                    Evaluating...
                  </>
                ) : (
                  'Score my sentence'
                )}
              </button>
            </div>
          </div>
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-xl flex items-start border border-red-100">
              <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Total Score Card */}
            <div className="bg-white rounded-2xl shadow-xl shadow-purple-100/50 p-8 text-center border border-gray-100 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-indigo-500 to-purple-500"></div>
              <h2 className="text-xl font-semibold text-gray-600 mb-2 uppercase tracking-widest">Total Score</h2>
              <div className="flex items-center justify-center">
                <span className={`text-7xl font-extrabold ${getScoreColor(result.total_score, 10).split(' ')[0]}`}>
                  {result.total_score}
                </span>
                <span className="text-3xl text-gray-400 font-medium ml-2">/ 10</span>
              </div>
              <p className="mt-6 text-gray-600 max-w-2xl mx-auto text-lg leading-relaxed bg-gray-50 p-4 rounded-xl inline-block border border-gray-100">
                {result.overall_feedback}
              </p>
            </div>

            {/* Breakdown Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Action Verb */}
              <div className={`rounded-2xl p-6 border-2 transition-all hover:-translate-y-1 ${getScoreColor(result.breakdown.action_verb.score, 4)}`}>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="font-bold text-lg flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5" />
                      Action Verb
                    </h3>
                    <span className="text-sm font-semibold opacity-75 mt-1 block uppercase tracking-wider">
                      {result.breakdown.action_verb.bloom_level || 'Unknown'} Level
                    </span>
                  </div>
                  <div className="text-2xl font-black bg-white/50 px-3 py-1 rounded-lg backdrop-blur-sm shadow-sm">
                    {result.breakdown.action_verb.score}/4
                  </div>
                </div>
                <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.action_verb.feedback}</p>
              </div>

              {/* Condition */}
              <div className={`rounded-2xl p-6 border-2 transition-all hover:-translate-y-1 ${getScoreColor(result.breakdown.condition.score, 2)}`}>
                <div className="flex justify-between items-start mb-4">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5" />
                    Condition
                  </h3>
                  <div className="text-2xl font-black bg-white/50 px-3 py-1 rounded-lg backdrop-blur-sm shadow-sm">
                    {result.breakdown.condition.score}/2
                  </div>
                </div>
                <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.condition.feedback}</p>
              </div>

              {/* Criterion */}
              <div className={`rounded-2xl p-6 border-2 transition-all hover:-translate-y-1 ${getScoreColor(result.breakdown.criterion.score, 2)}`}>
                <div className="flex justify-between items-start mb-4">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5" />
                    Criterion
                  </h3>
                  <div className="text-2xl font-black bg-white/50 px-3 py-1 rounded-lg backdrop-blur-sm shadow-sm">
                    {result.breakdown.criterion.score}/2
                  </div>
                </div>
                <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.criterion.feedback}</p>
              </div>

              {/* Clarity */}
              <div className={`rounded-2xl p-6 border-2 transition-all hover:-translate-y-1 ${getScoreColor(result.breakdown.clarity.score, 2)}`}>
                <div className="flex justify-between items-start mb-4">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5" />
                    Clarity
                  </h3>
                  <div className="text-2xl font-black bg-white/50 px-3 py-1 rounded-lg backdrop-blur-sm shadow-sm">
                    {result.breakdown.clarity.score}/2
                  </div>
                </div>
                <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.clarity.feedback}</p>
              </div>

            </div>

            {/* What it was supposed to be (Comparison Widget) */}
            {result.improved_objective && (
              <div className="bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl p-1 shadow-2xl relative mt-8">
                <div className="bg-white rounded-xl p-6 sm:p-8">
                  <h3 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-6 flex items-center gap-2">
                    <Sparkles className="w-7 h-7 text-indigo-500" />
                    What your text was supposed to be according to BLOOM'S TAXONOMY.
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Before */}
                    <div className="bg-gray-50 rounded-xl p-5 border border-gray-100 relative">
                      <div className="absolute -top-3 left-4 bg-gray-200 text-gray-600 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                        Your Original Text
                      </div>
                      <p className="text-gray-500 text-lg line-through decoration-red-400/50 mt-2">
                        {sentence}
                      </p>
                    </div>

                    {/* After */}
                    <div className="bg-indigo-50/50 rounded-xl p-5 border border-indigo-100 relative shadow-inner">
                      <div className="absolute -top-3 left-4 bg-indigo-600 text-white text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider shadow-sm flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" /> The 10/10 Version
                      </div>
                      <p className="text-indigo-900 text-lg font-medium mt-2">
                        {result.improved_objective}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
