import React, { useState } from 'react';
import axios from 'axios';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  GitMerge,
  Info,
  Lightbulb,
  Loader2,
  MinusCircle,
  Sparkles,
  Target,
} from 'lucide-react';

const CARD_TOOLTIPS = {
  action: "Checks whether the objective uses a clear, demonstrable action verb and maps it to Bloom's Taxonomy.",
  knowledge: 'Checks what specific topic or skill the action verb acts on, so the learning target is clear.',
  knowledgeDimension:
    'Classifies what kind of knowledge your objective targets - Factual (facts/terms), Conceptual (relationships/models), Procedural (methods/steps), or Metacognitive (self-reflection/strategy).',
  condition: 'Checks the method, tool, or circumstance students will use to demonstrate the objective.',
  criteria: 'Checks the measurable standard that shows how well students must perform.',
  assessment:
    "This maps your objective's action verb to Bloom's Taxonomy, then suggests assessment methods proven to test that cognitive level - based on standard Bloom's-aligned assessment practice.",
};

function Tooltip({ text }) {
  return (
    <span className="group relative inline-flex">
      <Info className="h-4 w-4 cursor-help opacity-65" />
      <span className="pointer-events-none absolute left-1/2 top-6 z-20 hidden w-64 -translate-x-1/2 rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs font-medium leading-snug text-gray-600 shadow-xl group-hover:block group-focus-within:block">
        {text}
      </span>
    </span>
  );
}

function PlaceholderCard({ title, tooltip, className = '' }) {
  return (
    <div className={`rounded-2xl border-2 border-gray-200 bg-gray-50 p-6 text-gray-400 ${className}`}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <h3 className="flex items-center gap-2 text-lg font-bold">
          <MinusCircle className="h-5 w-5" />
          {title}
          <Tooltip text={tooltip} />
        </h3>
      </div>
      <p className="text-sm font-medium leading-relaxed">Submit a sentence to see your results here</p>
    </div>
  );
}

function App() {
  const [sentence, setSentence] = useState('');
  const [condition, setCondition] = useState('');
  const [criteria, setCriteria] = useState('');
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
        sentence,
        condition: condition.trim() || null,
        criteria: criteria.trim() || null,
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while connecting to the server.');
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

  const formatBloomLevel = (level) => {
    if (Array.isArray(level)) return level.join(' + ');
    return level || 'Unknown';
  };

  const renderResults = () => {
    if (!result) {
      return (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
            <PlaceholderCard title="Action" tooltip={CARD_TOOLTIPS.action} />
            <PlaceholderCard title="Knowledge" tooltip={CARD_TOOLTIPS.knowledge} />
            <PlaceholderCard title="Condition" tooltip={CARD_TOOLTIPS.condition} />
            <PlaceholderCard title="Criteria" tooltip={CARD_TOOLTIPS.criteria} />
          </div>
          <PlaceholderCard
            title="Suggested Assessment Tools"
            tooltip={CARD_TOOLTIPS.assessment}
            className="min-h-44"
          />
        </div>
      );
    }

    const suggestion = result.suggested_assessment_tools;

    return (
      <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <div className="relative overflow-hidden rounded-2xl border border-gray-100 bg-white p-8 text-center shadow-xl shadow-purple-100/50">
          <div className="absolute left-0 top-0 h-2 w-full bg-gradient-to-r from-indigo-500 to-purple-500" />
          <h2 className="mb-2 text-xl font-semibold uppercase tracking-widest text-gray-600">Total Score</h2>
          <div className="flex items-center justify-center">
            <span className={`text-7xl font-extrabold ${getScoreColor(result.total_score, 10).split(' ')[0]}`}>
              {result.total_score}
            </span>
            <span className="ml-2 text-3xl font-medium text-gray-400">/ 10</span>
          </div>
          <p className="mx-auto mt-6 inline-block max-w-2xl rounded-xl border border-gray-100 bg-gray-50 p-4 text-lg leading-relaxed text-gray-600">
            {result.overall_feedback}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
          <div className={`rounded-2xl border-2 p-6 transition-all hover:-translate-y-1 ${getScoreColor(result.breakdown.action.score, 4)}`}>
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  <CheckCircle2 className="h-5 w-5" />
                  Action
                  <Tooltip text={CARD_TOOLTIPS.action} />
                </h3>
                <span className="mt-1 block text-sm font-semibold uppercase tracking-wider opacity-75">
                  {result.breakdown.action.bloom_level || 'Unknown'}
                </span>
                <span className="mt-0.5 block text-xs italic opacity-60">verb: "{result.breakdown.action.verb}"</span>
              </div>
              <div className="rounded-lg bg-white/50 px-3 py-1 text-2xl font-black shadow-sm backdrop-blur-sm">
                {result.breakdown.action.score}/4
              </div>
            </div>
            <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.action.feedback}</p>
          </div>

          <div className={`rounded-2xl border-2 p-6 transition-all hover:-translate-y-1 ${getScoreColor(result.breakdown.knowledge.score, 2)}`}>
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  <CheckCircle2 className="h-5 w-5" />
                  Knowledge
                  <Tooltip text={CARD_TOOLTIPS.knowledge} />
                </h3>
                {result.breakdown.knowledge.detected_knowledge && (
                  <span className="mt-0.5 block text-xs italic opacity-60">
                    "{result.breakdown.knowledge.detected_knowledge}"
                  </span>
                )}
              </div>
              <div className="rounded-lg bg-white/50 px-3 py-1 text-2xl font-black shadow-sm backdrop-blur-sm">
                <span className="inline-flex items-center gap-1">
                  {result.breakdown.knowledge.score}/2
                  {result.breakdown.knowledge.too_general && (
                    <AlertTriangle
                      className="h-4 w-4 text-amber-600"
                      aria-label="Knowledge target is too general to assess"
                    />
                  )}
                </span>
              </div>
            </div>
            <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.knowledge.feedback}</p>
            {(result.breakdown.knowledge.too_general || result.breakdown.knowledge.is_process_not_product || result.breakdown.knowledge.is_merged_cos) && (
              <div className="mt-4 flex flex-col items-start gap-2">
                {result.breakdown.knowledge.too_general && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-bold text-amber-800">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    Too general to assess
                  </span>
                )}
                {result.breakdown.knowledge.is_process_not_product && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-xs font-bold text-rose-800">
                    <AlertCircle className="h-3.5 w-3.5" />
                    Process, not product
                  </span>
                )}
                {result.breakdown.knowledge.is_merged_cos && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-xs font-bold text-violet-800">
                    <GitMerge className="h-3.5 w-3.5" />
                    Looks like two merged objectives
                  </span>
                )}
              </div>
            )}
            {result.breakdown.knowledge.knowledge_dimension?.length > 0 && (
              <div className="mt-4 flex flex-wrap items-center gap-2 text-sm font-medium">
                <span className="inline-flex items-center gap-1 opacity-80">
                  Knowledge type:
                  <Tooltip text={CARD_TOOLTIPS.knowledgeDimension} />
                </span>
                {result.breakdown.knowledge.knowledge_dimension.map((dimension) => (
                  <span
                    key={dimension}
                    className="rounded-full border border-teal-200 bg-white/70 px-3 py-1 text-sm font-semibold text-teal-800 shadow-sm"
                  >
                    {dimension}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div
            className={`rounded-2xl border-2 p-6 transition-all hover:-translate-y-1 ${
              result.breakdown.condition.detected
                ? getScoreColor(result.breakdown.condition.score, 2)
                : 'border-gray-200 bg-gray-50 text-gray-400'
            }`}
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  {result.breakdown.condition.detected ? <CheckCircle2 className="h-5 w-5" /> : <MinusCircle className="h-5 w-5" />}
                  Condition
                  <Tooltip text={CARD_TOOLTIPS.condition} />
                </h3>
                {result.breakdown.condition.detected && result.breakdown.condition.condition_text && (
                  <span className="mt-0.5 block text-xs italic opacity-60">
                    "{result.breakdown.condition.condition_text}"
                  </span>
                )}
                {!result.breakdown.condition.detected && (
                  <span className="mt-0.5 block text-xs italic opacity-50">Not provided (optional)</span>
                )}
              </div>
              <div className="rounded-lg bg-white/50 px-3 py-1 text-2xl font-black shadow-sm backdrop-blur-sm">
                {result.breakdown.condition.score}/2
              </div>
            </div>
            <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.condition.feedback}</p>
          </div>

          <div
            className={`rounded-2xl border-2 p-6 transition-all hover:-translate-y-1 ${
              result.breakdown.criteria.detected
                ? getScoreColor(result.breakdown.criteria.score, 2)
                : 'border-gray-200 bg-gray-50 text-gray-400'
            }`}
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="flex items-center gap-2 text-lg font-bold">
                  {result.breakdown.criteria.detected ? <CheckCircle2 className="h-5 w-5" /> : <MinusCircle className="h-5 w-5" />}
                  Criteria
                  <Tooltip text={CARD_TOOLTIPS.criteria} />
                </h3>
                {result.breakdown.criteria.detected && result.breakdown.criteria.criteria_text && (
                  <span className="mt-0.5 block text-xs italic opacity-60">
                    "{result.breakdown.criteria.criteria_text}"
                  </span>
                )}
                {!result.breakdown.criteria.detected && (
                  <span className="mt-0.5 block text-xs italic opacity-50">Not provided (optional)</span>
                )}
              </div>
              <div className="rounded-lg bg-white/50 px-3 py-1 text-2xl font-black shadow-sm backdrop-blur-sm">
                {result.breakdown.criteria.score}/2
              </div>
            </div>
            <p className="text-sm font-medium leading-relaxed opacity-90">{result.breakdown.criteria.feedback}</p>
          </div>
        </div>

        <div
          className={`min-h-44 rounded-2xl border-2 p-6 ${
            suggestion?.available
              ? 'border-teal-200 bg-gradient-to-br from-cyan-50 to-teal-50 text-teal-950'
              : 'border-gray-200 bg-gray-50 text-gray-400'
          }`}
        >
          <div className="mb-4 flex items-center gap-2">
            <Lightbulb className="h-5 w-5" />
            <h3 className="text-lg font-bold">Suggested Assessment Tools</h3>
            <Tooltip text={CARD_TOOLTIPS.assessment} />
          </div>

          {suggestion?.available ? (
            <>
              <p className="mb-4 text-xs font-bold uppercase tracking-wider text-teal-700">
                Bloom's Level: {formatBloomLevel(suggestion.bloom_level)}
              </p>
              <div className="mb-4 flex flex-wrap gap-2">
                {suggestion.tools?.map((tool) => (
                  <span
                    key={tool}
                    className="rounded-full border border-teal-200 bg-white/70 px-3 py-1 text-sm font-semibold text-teal-800 shadow-sm"
                  >
                    {tool}
                  </span>
                ))}
              </div>
              <p className="max-w-3xl text-sm font-medium leading-relaxed text-teal-800">{suggestion.note}</p>
            </>
          ) : (
            <p className="max-w-3xl text-sm font-medium leading-relaxed">
              {suggestion?.reason || 'Submit a sentence to see your results here'}
            </p>
          )}
        </div>

        {result.improved_objective && (
          <div className="relative mt-8 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-700 p-1 shadow-2xl">
            <div className="rounded-xl bg-white p-6 sm:p-8">
              <h3 className="mb-6 flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-2xl font-black text-transparent">
                <Sparkles className="h-7 w-7 text-indigo-500" />
                What your text was supposed to be according to BLOOM'S TAXONOMY.
              </h3>

              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                <div className="relative rounded-xl border border-gray-100 bg-gray-50 p-5">
                  <div className="absolute -top-3 left-4 rounded-full bg-gray-200 px-3 py-1 text-xs font-bold uppercase tracking-wider text-gray-600">
                    Your Original Text
                  </div>
                  <p className="mt-2 text-lg text-gray-500 line-through decoration-red-400/50">{sentence}</p>
                </div>

                <div className="relative rounded-xl border border-indigo-100 bg-indigo-50/50 p-5 shadow-inner">
                  <div className="absolute -top-3 left-4 flex items-center gap-1 rounded-full bg-indigo-600 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white shadow-sm">
                    <CheckCircle2 className="h-3 w-3" /> The 10/10 Version
                  </div>
                  <p className="mt-2 text-lg font-medium text-indigo-900">{result.improved_objective}</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-6 font-sans text-gray-800">
      <div className="mx-auto max-w-4xl space-y-8">
        <div className="space-y-4 pt-12 text-center">
          <div className="mb-4 inline-flex items-center justify-center rounded-full bg-indigo-100 p-3 shadow-inner">
            <Target className="h-10 w-10 text-indigo-600" />
          </div>
          <h1 className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-5xl font-extrabold tracking-tight text-transparent">
            Objective Sentence Scorer
          </h1>
          <p className="mx-auto max-w-2xl text-lg font-medium text-gray-500">
            Evaluate your learning objectives using Bloom's Taxonomy - Action, Knowledge, Condition &amp; Criteria.
          </p>
        </div>

        <div className="rounded-2xl border border-gray-100 bg-white p-8 shadow-xl shadow-indigo-100/50 transition-all duration-300 hover:shadow-2xl hover:shadow-indigo-100/60">
          <div className="space-y-5">
            <div>
              <label htmlFor="sentence" className="block text-sm font-semibold uppercase tracking-wider text-gray-700">
                Learning Objective (Action + Knowledge)
              </label>
              <textarea
                id="sentence"
                rows="3"
                className="mt-2 w-full rounded-xl border-gray-200 bg-gray-50/50 p-4 text-lg shadow-sm transition-colors placeholder:text-gray-400 focus:border-indigo-500 focus:ring-indigo-500"
                placeholder='e.g. "Determine the root of the given equation"'
                value={sentence}
                onChange={(e) => setSentence(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleScore();
                  }
                }}
              />
              <p className="mt-2 text-xs italic text-gray-400">
                Tip: try including what tool/method you'll use and how success is measured - e.g. "...using Python, with 90% accuracy."
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <label htmlFor="condition" className="block text-xs font-medium uppercase tracking-wider text-gray-500">
                  Condition <span className="text-gray-400 normal-case">(optional)</span>
                </label>
                <input
                  id="condition"
                  type="text"
                  className="mt-1 w-full rounded-lg border-gray-200 bg-gray-50/50 p-3 text-sm shadow-sm transition-colors placeholder:text-gray-400 focus:border-indigo-400 focus:ring-indigo-400"
                  placeholder='e.g. "using Newton-Raphson method through C++"'
                  value={condition}
                  onChange={(e) => setCondition(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="criteria" className="block text-xs font-medium uppercase tracking-wider text-gray-500">
                  Criteria <span className="text-gray-400 normal-case">(optional)</span>
                </label>
                <input
                  id="criteria"
                  type="text"
                  className="mt-1 w-full rounded-lg border-gray-200 bg-gray-50/50 p-3 text-sm shadow-sm transition-colors placeholder:text-gray-400 focus:border-indigo-400 focus:ring-indigo-400"
                  placeholder='e.g. "accurate to second decimal place"'
                  value={criteria}
                  onChange={(e) => setCriteria(e.target.value)}
                />
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={handleScore}
                disabled={loading || !sentence.trim()}
                className="inline-flex items-center rounded-xl border border-transparent bg-indigo-600 px-6 py-3 text-base font-medium text-white shadow-sm transition-all active:scale-95 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2 className="-ml-1 mr-2 h-5 w-5 animate-spin" />
                    Evaluating...
                  </>
                ) : (
                  'Score my sentence'
                )}
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 flex items-start rounded-xl border border-red-100 bg-red-50 p-4 text-red-700">
              <AlertCircle className="mr-2 mt-0.5 h-5 w-5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {renderResults()}
      </div>
    </div>
  );
}

export default App;
