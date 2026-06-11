import { useState, useRef, useCallback, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

/* ── Theme Definitions ── */
type ThemeKey = 'light' | 'dark' | 'ocean' | 'sunset' | 'forest' | 'lavender';

interface Theme {
  label: string;
  preview: string; // small color dot
  bg: string;
  headerBg: string;
  headerText: string;
  headerAccent: string;
  card: string;
  cardBorder: string;
  text: string;
  textSecondary: string;
  textMuted: string;
  input: string;
  inputBorder: string;
  inputText: string;
  gaugeTrack: string;
  statBg: string;
  footerBg: string;
  footerBorder: string;
}

const THEMES: Record<ThemeKey, Theme> = {
  light: {
    label: 'Light',
    preview: '#f8fafc',
    bg: 'bg-slate-50',
    headerBg: 'bg-gradient-to-r from-brand-900 via-brand-800 to-brand-700',
    headerText: 'text-white',
    headerAccent: 'text-brand-200',
    card: 'bg-white/80 backdrop-blur-sm',
    cardBorder: 'border-gray-200',
    text: 'text-gray-900',
    textSecondary: 'text-gray-600',
    textMuted: 'text-gray-400',
    input: 'bg-white',
    inputBorder: 'border-gray-200',
    inputText: 'text-gray-700',
    gaugeTrack: '#e5e7eb',
    statBg: 'bg-gray-50',
    footerBg: 'bg-white',
    footerBorder: 'border-gray-200',
  },
  dark: {
    label: 'Dark',
    preview: '#0f172a',
    bg: 'bg-slate-900',
    headerBg: 'bg-gradient-to-r from-slate-950 via-slate-900 to-slate-800',
    headerText: 'text-white',
    headerAccent: 'text-slate-400',
    card: 'bg-slate-800/80 backdrop-blur-sm',
    cardBorder: 'border-slate-700',
    text: 'text-white',
    textSecondary: 'text-slate-300',
    textMuted: 'text-slate-500',
    input: 'bg-slate-700',
    inputBorder: 'border-slate-600',
    inputText: 'text-slate-100',
    gaugeTrack: '#334155',
    statBg: 'bg-slate-700/50',
    footerBg: 'bg-slate-800',
    footerBorder: 'border-slate-700',
  },
  ocean: {
    label: 'Ocean',
    preview: '#0c4a6e',
    bg: 'bg-sky-50',
    headerBg: 'bg-gradient-to-r from-sky-900 via-cyan-800 to-teal-700',
    headerText: 'text-white',
    headerAccent: 'text-sky-200',
    card: 'bg-white/80 backdrop-blur-sm',
    cardBorder: 'border-sky-200',
    text: 'text-sky-950',
    textSecondary: 'text-sky-700',
    textMuted: 'text-sky-400',
    input: 'bg-white',
    inputBorder: 'border-sky-200',
    inputText: 'text-sky-900',
    gaugeTrack: '#bae6fd',
    statBg: 'bg-sky-50',
    footerBg: 'bg-white',
    footerBorder: 'border-sky-200',
  },
  sunset: {
    label: 'Sunset',
    preview: '#7c2d12',
    bg: 'bg-orange-50',
    headerBg: 'bg-gradient-to-r from-orange-900 via-red-800 to-rose-700',
    headerText: 'text-white',
    headerAccent: 'text-orange-200',
    card: 'bg-white/80 backdrop-blur-sm',
    cardBorder: 'border-orange-200',
    text: 'text-orange-950',
    textSecondary: 'text-orange-700',
    textMuted: 'text-orange-400',
    input: 'bg-white',
    inputBorder: 'border-orange-200',
    inputText: 'text-orange-900',
    gaugeTrack: '#fed7aa',
    statBg: 'bg-orange-50',
    footerBg: 'bg-white',
    footerBorder: 'border-orange-200',
  },
  forest: {
    label: 'Forest',
    preview: '#14532d',
    bg: 'bg-emerald-50',
    headerBg: 'bg-gradient-to-r from-emerald-900 via-green-800 to-teal-700',
    headerText: 'text-white',
    headerAccent: 'text-emerald-200',
    card: 'bg-white/80 backdrop-blur-sm',
    cardBorder: 'border-emerald-200',
    text: 'text-emerald-950',
    textSecondary: 'text-emerald-700',
    textMuted: 'text-emerald-400',
    input: 'bg-white',
    inputBorder: 'border-emerald-200',
    inputText: 'text-emerald-900',
    gaugeTrack: '#a7f3d0',
    statBg: 'bg-emerald-50',
    footerBg: 'bg-white',
    footerBorder: 'border-emerald-200',
  },
  lavender: {
    label: 'Lavender',
    preview: '#581c87',
    bg: 'bg-purple-50',
    headerBg: 'bg-gradient-to-r from-purple-900 via-violet-800 to-fuchsia-700',
    headerText: 'text-white',
    headerAccent: 'text-purple-200',
    card: 'bg-white/80 backdrop-blur-sm',
    cardBorder: 'border-purple-200',
    text: 'text-purple-950',
    textSecondary: 'text-purple-700',
    textMuted: 'text-purple-400',
    input: 'bg-white',
    inputBorder: 'border-purple-200',
    inputText: 'text-purple-900',
    gaugeTrack: '#e9d5ff',
    statBg: 'bg-purple-50',
    footerBg: 'bg-white',
    footerBorder: 'border-purple-200',
  },
};

interface AnalysisResult {
  ai_probability: number;
  classification: string;
  confidence: string;
  text_length: number;
  word_count: number;
  sentence_count: number;
  features: Record<string, number>;
  patterns: string[];
  ml_model_used: boolean;
}

interface FactCheckResult {
  total_claims: number;
  verified: number;
  unverified: number;
  claims: Array<{
    claim: string;
    status: string;
    summary: string;
    sources: string[];
  }>;
}

interface QualityResult {
  readability: {
    flesch_reading_ease: number;
    flesch_kincaid_grade: number;
    gunning_fog: number;
    automated_readability_index: number;
    coleman_liau_index: number;
    smog_index: number;
  };
  language: {
    language: string;
    language_name: string;
    confidence: string;
  };
  writing_style: {
    vocabulary_richness: number;
    avg_word_length: number;
    avg_sentence_length: number;
    writing_style_score: number;
  };
  plagiarism_check: {
    similarity_risk: string;
    vocabulary_uniqueness: number;
    recommendation: string;
  };
  text_length: number;
  word_count: number;
}

interface BatchResult {
  total_files: number;
  analyzed_count: number;
  results: Array<{
    filename: string;
    ai_probability?: number;
    classification?: string;
    word_count?: number;
    error?: string;
  }>;
  summary: {
    average_ai_probability: number;
    high_risk_count: number;
    low_risk_count: number;
  };
}

/* ── Circular Gauge Component ── */
function CircularGauge({ value, size = 180, trackColor = '#e5e7eb' }: { value: number; size?: number; trackColor?: string }) {
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const color = value >= 70 ? '#ef4444' : value >= 50 ? '#f59e0b' : '#22c55e';

  return (
    <svg width={size} height={size} viewBox="0 0 160 160" className="mx-auto">
      {/* Background ring */}
      <circle cx="80" cy="80" r={radius} fill="none" stroke={trackColor} strokeWidth="12" />
      {/* Colored arc */}
      <circle
        cx="80" cy="80" r={radius}
        fill="none" stroke={color} strokeWidth="12"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        className="gauge-circle"
        transform="rotate(-90 80 80)"
      />
      {/* Center text */}
      <text x="80" y="72" textAnchor="middle" className="text-4xl font-bold" fill={color} fontSize="36" fontWeight="700">
        {value}%
      </text>
      <text x="80" y="96" textAnchor="middle" fill="#6b7280" fontSize="11" fontWeight="500">
        AI Probability
      </text>
    </svg>
  );
}

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<'analyze' | 'factcheck' | 'quality'>('analyze');
  const [factChecking, setFactChecking] = useState(false);
  const [factResult, setFactResult] = useState<FactCheckResult | null>(null);
  const [qualityResult, setQualityResult] = useState<QualityResult | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(false);
  const [showBatch, setShowBatch] = useState(false);
  const [themeKey, setThemeKey] = useState<ThemeKey>(() => (localStorage.getItem('aw-theme') as ThemeKey) || 'light');
  const [showThemePicker, setShowThemePicker] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const themeRef = useRef<HTMLDivElement>(null);

  const t = THEMES[themeKey];

  // Persist theme
  useEffect(() => { localStorage.setItem('aw-theme', themeKey); }, [themeKey]);

  // Close theme picker on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (themeRef.current && !themeRef.current.contains(e.target as Node)) setShowThemePicker(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const analyzeText = async (textToAnalyze: string) => {
    if (!textToAnalyze.trim()) { setError('Please enter text to analyze'); return; }
    if (textToAnalyze.trim().length < 50) { setError('Text must be at least 50 characters'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToAnalyze, privacy_mode: privacyMode })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Analysis failed');
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze text. Make sure the backend is running.');
    } finally { setLoading(false); }
  };

  const handleAnalyze = () => analyzeText(text);

  const handleFileUpload = async (file: File) => {
    setUploading(true); setUploadProgress(30); setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      setUploadProgress(50);
      const response = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await response.json();
      setUploadProgress(80);
      if (!response.ok) throw new Error(data.error || 'File upload failed');
      setText(data.extracted_text);
      setUploadProgress(100);
      await analyzeText(data.extracted_text);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload file');
    } finally { setUploading(false); setUploadProgress(0); }
  };

  const handleLoadSample = async (type: 'ai' | 'human') => {
    setLoading(true); setError('');
    try {
      const response = await fetch(`/api/sample-text?type=${type}`);
      if (!response.ok) { const d = await response.json(); throw new Error(d.error || 'Failed'); }
      const data = await response.json();
      setText(data.text);
    } catch { setError('Failed to load sample text'); } finally { setLoading(false); }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) handleFileUpload(e.target.files[0]);
  };

  const handleFactCheck = async () => {
    if (!text.trim()) { setError('Please enter text to fact-check'); return; }
    setFactChecking(true); setError('');
    try {
      const response = await fetch('/api/fact-check', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, privacy_mode: privacyMode })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Fact-check failed');
      setFactResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fact-check');
    } finally { setFactChecking(false); }
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    try {
      const response = await fetch('/api/generate-report', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis: result })
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'authenticwrite-report.pdf';
      document.body.appendChild(a); a.click();
      window.URL.revokeObjectURL(url); document.body.removeChild(a);
    } catch { setError('Failed to generate report'); }
  };

  const handleQualityAnalysis = async () => {
    if (!text.trim()) { setError('Please enter text to analyze'); return; }
    setQualityLoading(true); setError('');
    try {
      const response = await fetch('/api/quality-analysis', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, privacy_mode: privacyMode })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Quality analysis failed');
      setQualityResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze quality');
    } finally { setQualityLoading(false); }
  };

  const handleBatchUpload = async (files: FileList) => {
    if (files.length === 0) return;
    if (files.length > 20) { setError('Maximum 20 files allowed'); return; }
    setBatchLoading(true); setError(''); setBatchResult(null);
    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) formData.append('files', files[i]);
      const response = await fetch('/api/batch-analyze', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Batch analysis failed');
      setBatchResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze batch');
    } finally { setBatchLoading(false); }
  };

  const getProbColor = (p: number) => p >= 70 ? '#ef4444' : p >= 50 ? '#f59e0b' : '#22c55e';
  const getProbBg = (p: number) => p >= 70 ? 'bg-red-50 border-red-200' : p >= 50 ? 'bg-amber-50 border-amber-200' : 'bg-green-50 border-green-200';

  const getFeatureChartData = () => {
    if (!result) return [];
    const f = result.features;
    return [
      { name: 'Sent. Length', value: f.avg_sentence_length || 0 },
      { name: 'Word Length', value: f.avg_word_length || 0 },
      { name: 'Vocabulary', value: (f.vocabulary_diversity || 0) * 100 },
      { name: 'Punctuation', value: (f.punctuation_ratio || 0) * 100 },
      { name: 'Stop Words', value: (f.stop_word_ratio || 0) * 100 },
      { name: 'Long Words', value: (f.long_word_ratio || 0) * 100 },
    ];
  };

  const getPieData = () => {
    if (!result) return [];
    return [
      { name: 'AI-Generated', value: result.ai_probability, color: '#ef4444' },
      { name: 'Human-Written', value: 100 - result.ai_probability, color: '#22c55e' },
    ];
  };

  const wordCount = text.split(/\s+/).filter(Boolean).length;

  return (
    <div className={`min-h-screen ${t.bg} transition-colors duration-300`}>

      {/* ═══ HEADER ═══ */}
      <header className={`${t.headerBg} ${t.headerText} sticky top-0 z-50 shadow-lg transition-colors duration-300`}>
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-white/20 flex items-center justify-center text-lg font-bold">
                AW
              </div>
              <div>
                <h1 className="text-lg font-bold leading-tight">AuthentiWrite</h1>
                <p className={`text-[10px] ${t.headerAccent} leading-tight`}>AI Academic Integrity Detection</p>
              </div>
            </div>

            {/* Nav Tabs */}
            <nav className="hidden sm:flex items-center gap-1">
              {[
                { key: 'analyze', label: 'AI Detection', icon: '🔍' },
                { key: 'factcheck', label: 'Fact Check', icon: '✓' },
                { key: 'quality', label: 'Quality', icon: '📊' },
              ].map(tab => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as typeof activeTab)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeTab === tab.key
                      ? 'bg-white/20 text-white'
                      : `${t.headerAccent} hover:text-white hover:bg-white/10`
                  }`}
                >
                  <span className="mr-1.5">{tab.icon}</span>{tab.label}
                </button>
              ))}
            </nav>

            {/* Theme Picker + Privacy */}
            <div className="flex items-center gap-3">
              {/* Theme Picker */}
              <div className="relative" ref={themeRef}>
                <button
                  onClick={() => setShowThemePicker(!showThemePicker)}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors text-sm"
                  title="Change theme"
                >
                  <div className="w-4 h-4 rounded-full border-2 border-white/40" style={{ background: t.preview }} />
                  <svg className="w-3 h-3 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {showThemePicker && (
                  <div className="absolute right-0 top-full mt-2 w-44 bg-white rounded-xl shadow-xl border border-gray-200 p-2 z-50 animate-slide-up">
                    <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider px-2 mb-1">Theme</p>
                    {(Object.keys(THEMES) as ThemeKey[]).map(key => (
                      <button
                        key={key}
                        onClick={() => { setThemeKey(key); setShowThemePicker(false); }}
                        className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors ${
                          themeKey === key ? 'bg-brand-50 text-brand-700 font-semibold' : 'text-gray-600 hover:bg-gray-50'
                        }`}
                      >
                        <div className="w-5 h-5 rounded-full border-2 border-gray-200 shrink-0" style={{ background: THEMES[key].preview }} />
                        {THEMES[key].label}
                        {themeKey === key && (
                          <svg className="w-4 h-4 ml-auto text-brand-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Privacy Toggle */}
              <label className="flex items-center gap-2 cursor-pointer text-sm">
                <div className={`w-9 h-5 rounded-full transition-colors relative ${privacyMode ? 'bg-green-400' : 'bg-white/20'}`}
                  onClick={() => setPrivacyMode(!privacyMode)}>
                  <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${privacyMode ? 'translate-x-4' : 'translate-x-0.5'}`} />
                </div>
                <span className="hidden md:inline text-brand-200 text-xs">Privacy</span>
              </label>
            </div>
          </div>

          {/* Mobile tabs */}
          <div className="sm:hidden flex gap-1 pb-2">
            {[
              { key: 'analyze', label: 'Detect' },
              { key: 'factcheck', label: 'Facts' },
              { key: 'quality', label: 'Quality' },
            ].map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as typeof activeTab)}
                className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  activeTab === tab.key ? 'bg-white/20 text-white' : `${t.headerAccent} hover:text-white`
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6">

        {/* ═══ HERO / INPUT SECTION ═══ */}
        <section className="mb-6 animate-fade-in">
          <div className="text-center mb-5">
            <h2 className={`text-2xl sm:text-3xl font-bold ${t.text} mb-2`}>
              Detect AI-Generated Text Instantly
            </h2>
            <p className={`${t.textSecondary} text-lg`}>
              Paste text, upload a file, or try a sample — powered by ensemble ML with 96% accuracy
            </p>
          </div>

          <div className={`${t.card} border ${t.cardBorder} rounded-2xl shadow-lg p-5 sm:p-6 transition-colors duration-300`}>
            {/* File Upload Zone */}
            <div
              className={`border-2 border-dashed rounded-xl p-5 text-center transition-all cursor-pointer mb-4 ${
                dragActive
                  ? 'border-brand-500 bg-brand-50 scale-[1.01]'
                  : 'border-gray-200 hover:border-brand-300 hover:bg-brand-50/30'
              }`}
              onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileInput}
                accept=".pdf,.docx,.doc,.txt,.md,.rtf,.odt,.tex,.xlsx,.xls,.csv,.json,.py,.js,.html,.css,.java,.cpp,.cs,.php,.rb,.ts,.sh" />
              {uploading ? (
                <div className="py-2">
                  <div className="w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                  <p className="text-sm text-gray-500">Processing... {uploadProgress}%</p>
                  <div className="w-48 h-1.5 bg-gray-200 rounded-full mx-auto mt-2">
                    <div className="h-full bg-brand-500 rounded-full transition-all" style={{width: `${uploadProgress}%`}} />
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-1">
                  <div className="w-10 h-10 rounded-full bg-brand-100 flex items-center justify-center mb-1">
                    <svg className="w-5 h-5 text-brand-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <p className="text-sm text-gray-600">
                    <span className="font-semibold text-brand-600">Click to upload</span> or drag & drop
                  </p>
                  <p className="text-xs text-gray-400">PDF, Word, Excel, Text, Code files</p>
                </div>
              )}
            </div>

            {/* Text Input */}
            <div className="relative">
              <textarea
                className={`w-full h-44 p-4 pr-4 pb-10 border ${t.inputBorder} rounded-xl focus:ring-2 focus:ring-brand-500 focus:border-transparent resize-none ${t.inputText} text-sm leading-relaxed placeholder-gray-400 ${t.input} transition-colors duration-300`}
                placeholder="Paste or type your text here (minimum 50 characters)..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              {/* Bottom bar inside textarea */}
              <div className={`absolute bottom-0 left-0 right-0 px-4 py-2 flex items-center justify-between rounded-b-xl`}>
                <span className={`text-xs ${t.textMuted}`}>
                  {text.length} chars &middot; {wordCount} words
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setText('')}
                    className={`px-3 py-1 text-xs text-gray-400 hover:text-gray-600 transition-colors ${!text ? 'invisible' : ''}`}
                  >
                    Clear
                  </button>
                  <button
                    onClick={handleAnalyze}
                    disabled={loading || !text.trim()}
                    className="px-5 py-2 bg-brand-600 text-white text-sm font-semibold rounded-lg hover:bg-brand-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow-md"
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Analyzing...
                      </span>
                    ) : 'Analyze Text'}
                  </button>
                </div>
              </div>
            </div>

            {/* Action buttons row */}
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <button
                onClick={() => handleLoadSample('ai')}
                disabled={loading}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-red-50 text-red-600 border border-red-200 hover:bg-red-100 transition-colors disabled:opacity-50"
              >
                Try AI Sample
              </button>
              <button
                onClick={() => handleLoadSample('human')}
                disabled={loading}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-green-50 text-green-600 border border-green-200 hover:bg-green-100 transition-colors disabled:opacity-50"
              >
                Try Human Sample
              </button>
              <div className="flex-1" />
              <button
                onClick={() => setShowBatch(!showBatch)}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-50 text-purple-600 border border-purple-200 hover:bg-purple-100 transition-colors"
              >
                {showBatch ? 'Hide' : 'Batch'} Upload
              </button>
            </div>

            {/* Batch Upload (collapsible) */}
            {showBatch && (
              <div className="mt-3 p-4 bg-purple-50 rounded-xl border border-purple-200 animate-slide-up">
                <p className="text-sm font-semibold text-purple-800 mb-2">Batch Processing (max 20 files)</p>
                <input type="file" multiple
                  onChange={(e) => e.target.files && handleBatchUpload(e.target.files)}
                  accept=".pdf,.docx,.doc,.txt,.md"
                  className="block w-full text-xs text-purple-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-purple-100 file:text-purple-700 hover:file:bg-purple-200"
                />
                {batchLoading && (
                  <div className="mt-3 flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-purple-600 border-t-transparent rounded-full animate-spin" />
                    <span className="text-xs text-purple-600">Analyzing files...</span>
                  </div>
                )}
                {batchResult && !batchLoading && (
                  <div className="mt-3 p-3 bg-white rounded-lg">
                    <div className="grid grid-cols-3 gap-2 text-center mb-2">
                      <div>
                        <div className="text-lg font-bold text-gray-800">{batchResult.total_files}</div>
                        <div className="text-[10px] text-gray-500">Files</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold text-purple-600">{batchResult.summary.average_ai_probability}%</div>
                        <div className="text-[10px] text-gray-500">Avg AI</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold text-red-600">{batchResult.summary.high_risk_count}</div>
                        <div className="text-[10px] text-gray-500">High Risk</div>
                      </div>
                    </div>
                    <div className="max-h-28 overflow-y-auto text-xs space-y-1">
                      {batchResult.results.map((r, i) => (
                        <div key={i} className="flex justify-between py-1 px-2 rounded bg-gray-50">
                          <span className="text-gray-600 truncate flex-1 mr-2">{r.filename}</span>
                          {r.error ? (
                            <span className="text-red-500">{r.error}</span>
                          ) : (
                            <span className={`font-medium ${(r.ai_probability || 0) >= 70 ? 'text-red-600' : 'text-green-600'}`}>
                              {r.ai_probability}%
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-xl flex items-center gap-2 animate-slide-up">
              <svg className="w-5 h-5 text-red-500 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          )}
        </section>

        {/* ═══ RESULTS SECTION ═══ */}
        {/* AI Detection Tab */}
        {activeTab === 'analyze' && (
          <>
            {loading && (
              <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
                <div className="w-16 h-16 border-4 border-brand-600 border-t-transparent rounded-full animate-spin mb-4" />
                <p className={`${t.textSecondary} font-medium`}>Analyzing your text...</p>
                <p className={`${t.textMuted} text-sm mt-1`}>Running ensemble ML model</p>
              </div>
            )}

            {result && !loading && (
              <section className="animate-slide-up space-y-5">
                {/* Score + Classification Banner */}
                <div className={`rounded-2xl border p-6 ${getProbBg(result.ai_probability)}`}>
                  <div className="flex flex-col md:flex-row items-center gap-6">
                    {/* Gauge */}
                    <CircularGauge value={result.ai_probability} trackColor={t.gaugeTrack} />

                    {/* Classification */}
                    <div className="flex-1 text-center md:text-left">
                      <h3 className="text-2xl font-bold" style={{ color: getProbColor(result.ai_probability) }}>
                        {result.classification}
                      </h3>
                      <p className={`${t.textMuted} text-sm mt-1`}>
                        Confidence: <span className={`font-semibold ${t.text}`}>{result.confidence}</span>
                      </p>
                      <div className="flex flex-wrap gap-2 mt-3 justify-center md:justify-start">
                        {result.patterns.map((p, i) => (
                          <span key={i} className="px-2.5 py-1 text-xs font-medium bg-amber-100 text-amber-800 rounded-full">
                            {p}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-2">
                      <button
                        onClick={handleDownloadReport}
                        className="px-4 py-2 bg-brand-600 text-white text-sm font-medium rounded-lg hover:bg-brand-700 transition-colors shadow-sm flex items-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        PDF Report
                      </button>
                      <span className="text-[10px] text-gray-400 text-center">
                        {result.ml_model_used ? 'ML + Rules' : 'Rule-based'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { label: 'Words', value: result.word_count, icon: '📝', color: 'brand' },
                    { label: 'Sentences', value: result.sentence_count, icon: '📄', color: 'brand' },
                    { label: 'Characters', value: result.text_length, icon: '🔤', color: 'brand' },
                  ].map(s => (
                    <div key={s.label} className={`${t.card} border ${t.cardBorder} rounded-xl p-4 text-center hover-lift shadow-sm transition-colors duration-300`}>
                      <div className="text-2xl mb-1">{s.icon}</div>
                      <div className={`text-2xl font-bold ${t.text}`}>{s.value.toLocaleString()}</div>
                      <div className={`text-xs ${t.textMuted}`}>{s.label}</div>
                    </div>
                  ))}
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {/* Pie Chart */}
                  <div className={`${t.card} border ${t.cardBorder} rounded-xl p-5 shadow-sm transition-colors duration-300`}>
                    <h4 className={`text-sm font-semibold ${t.textSecondary} mb-3`}>Human vs AI Distribution</h4>
                    <div className="h-52">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={getPieData()} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3} dataKey="value">
                            {getPieData().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="flex justify-center gap-4 mt-2">
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                        <span className="text-xs text-gray-500">Human</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-red-500" />
                        <span className="text-xs text-gray-500">AI</span>
                      </div>
                    </div>
                  </div>

                  {/* Bar Chart */}
                  <div className={`${t.card} border ${t.cardBorder} rounded-xl p-5 shadow-sm transition-colors duration-300`}>
                    <h4 className={`text-sm font-semibold ${t.textSecondary} mb-3`}>Stylometric Features</h4>
                    <div className="h-52">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getFeatureChartData()}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Bar dataKey="value" fill="#6366f1" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </section>
            )}

            {!result && !loading && (
              <div className={`text-center py-16 ${t.textMuted}`}>
                <svg className="w-20 h-20 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className={`${t.textMuted} text-lg font-medium`}>Enter text above to get started</p>
                <p className={`${t.textMuted} text-sm mt-1`}>Results will appear here after analysis</p>
              </div>
            )}
          </>
        )}

        {/* ═══ FACT CHECK TAB ═══ */}
        {activeTab === 'factcheck' && (
          <section className="animate-fade-in">
            <div className={`${t.card} border ${t.cardBorder} rounded-2xl shadow-sm p-6 transition-colors duration-300`}>
              <div className="text-center mb-5">
                <h3 className={`text-xl font-bold ${t.text} mb-1`}>Fact Verification</h3>
                <p className={`${t.textMuted} text-sm`}>Verify factual claims found in the text</p>
                <button
                  onClick={handleFactCheck}
                  disabled={factChecking || !text.trim()}
                  className="mt-3 px-6 py-2.5 bg-brand-600 text-white font-semibold text-sm rounded-lg hover:bg-brand-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all shadow-sm"
                >
                  {factChecking ? (
                    <span className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Checking...
                    </span>
                  ) : 'Check Facts'}
                </button>
              </div>

              {factResult && !factChecking && (
                <div className="space-y-4 animate-slide-up">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-4 bg-green-50 rounded-xl text-center border border-green-200">
                      <div className="text-3xl font-bold text-green-600">{factResult.verified}</div>
                      <div className="text-xs text-green-500 font-medium">Verified</div>
                    </div>
                    <div className="p-4 bg-amber-50 rounded-xl text-center border border-amber-200">
                      <div className="text-3xl font-bold text-amber-600">{factResult.unverified}</div>
                      <div className="text-xs text-amber-500 font-medium">Unverified</div>
                    </div>
                  </div>

                  {factResult.claims.length === 0 ? (
                    <p className="text-center text-gray-400 py-6">No factual claims detected</p>
                  ) : (
                    <div className="space-y-3">
                      {factResult.claims.map((claim, i) => (
                        <div key={i} className={`p-4 rounded-xl border ${claim.status === 'verified' ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
                          <span className={`inline-block px-2 py-0.5 text-[10px] font-semibold rounded-full mb-2 ${
                            claim.status === 'verified' ? 'bg-green-200 text-green-800' : 'bg-amber-200 text-amber-800'
                          }`}>
                            {claim.status === 'verified' ? 'VERIFIED' : 'UNVERIFIED'}
                          </span>
                          <p className="text-sm text-gray-800 mb-1">{claim.claim}</p>
                          <p className="text-xs text-gray-500">{claim.summary}</p>
                          {claim.sources.length > 0 && (
                            <p className="text-[10px] text-gray-400 mt-1">Sources: {claim.sources.join(', ')}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {!factResult && !factChecking && (
                <div className="text-center py-10 text-gray-300">
                  <svg className="w-16 h-16 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-gray-400">Click "Check Facts" to verify claims</p>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ═══ QUALITY TAB ═══ */}
        {activeTab === 'quality' && (
          <section className="animate-fade-in">
            <div className={`${t.card} border ${t.cardBorder} rounded-2xl shadow-sm p-6 transition-colors duration-300`}>
              <div className="text-center mb-5">
                <h3 className={`text-xl font-bold ${t.text} mb-1`}>Writing Quality Analysis</h3>
                <p className={`${t.textMuted} text-sm`}>Readability, style, and plagiarism assessment</p>
                <button
                  onClick={handleQualityAnalysis}
                  disabled={qualityLoading || !text.trim()}
                  className="mt-3 px-6 py-2.5 bg-brand-600 text-white font-semibold text-sm rounded-lg hover:bg-brand-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all shadow-sm"
                >
                  {qualityLoading ? (
                    <span className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Analyzing...
                    </span>
                  ) : 'Analyze Quality'}
                </button>
              </div>

              {qualityResult && !qualityLoading && (
                <div className="space-y-5 animate-slide-up">
                  {/* Language */}
                  <div className="p-4 bg-blue-50 rounded-xl border border-blue-200 flex items-center justify-between">
                    <div>
                      <p className="text-xs text-blue-500 font-medium">Detected Language</p>
                      <p className="text-xl font-bold text-blue-700">{qualityResult.language.language_name}</p>
                    </div>
                    <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${
                      qualityResult.language.confidence === 'high' ? 'bg-green-100 text-green-700' :
                      qualityResult.language.confidence === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {qualityResult.language.confidence}
                    </span>
                  </div>

                  {/* Readability Grid */}
                  <div>
                    <h4 className={`text-sm font-semibold ${t.textSecondary} mb-3`}>Readability Scores</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {[
                        { label: 'Flesch Ease', val: qualityResult.readability.flesch_reading_ease, sub: '0-100' },
                        { label: 'Kincaid Grade', val: qualityResult.readability.flesch_kincaid_grade, sub: 'Grade' },
                        { label: 'Gunning Fog', val: qualityResult.readability.gunning_fog, sub: 'Years' },
                        { label: 'ARI', val: qualityResult.readability.automated_readability_index, sub: 'Grade' },
                        { label: 'Coleman-Liau', val: qualityResult.readability.coleman_liau_index, sub: 'Grade' },
                        { label: 'SMOG', val: qualityResult.readability.smog_index, sub: 'Grade' },
                      ].map(m => (
                        <div key={m.label} className={`p-3 ${t.statBg} rounded-xl text-center hover-lift`}>
                          <div className="text-lg font-bold text-brand-600">{m.val}</div>
                          <div className={`text-xs ${t.textSecondary} font-medium`}>{m.label}</div>
                          <div className={`text-[10px] ${t.textMuted}`}>{m.sub}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Writing Style */}
                  <div>
                    <h4 className={`text-sm font-semibold ${t.textSecondary} mb-3`}>Writing Style</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {[
                        { label: 'Vocabulary', val: `${(qualityResult.writing_style.vocabulary_richness * 100).toFixed(1)}%` },
                        { label: 'Word Length', val: qualityResult.writing_style.avg_word_length },
                        { label: 'Sent. Length', val: qualityResult.writing_style.avg_sentence_length },
                        { label: 'Style Score', val: qualityResult.writing_style.writing_style_score },
                      ].map(m => (
                        <div key={m.label} className={`p-3 ${t.statBg} rounded-xl text-center hover-lift`}>
                          <div className="text-lg font-bold text-green-600">{m.val}</div>
                          <div className={`text-xs ${t.textSecondary} font-medium`}>{m.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Plagiarism */}
                  <div className={`p-4 rounded-xl border ${
                    qualityResult.plagiarism_check.similarity_risk === 'low' ? 'bg-green-50 border-green-200' :
                    qualityResult.plagiarism_check.similarity_risk === 'medium' ? 'bg-yellow-50 border-yellow-200' :
                    'bg-red-50 border-red-200'
                  }`}>
                    <div className="flex items-center justify-between mb-2">
                      <h4 className={`text-sm font-semibold ${t.textSecondary}`}>Plagiarism Check</h4>
                      <span className={`px-2.5 py-1 text-xs font-semibold rounded-full ${
                        qualityResult.plagiarism_check.similarity_risk === 'low' ? 'bg-green-200 text-green-800' :
                        qualityResult.plagiarism_check.similarity_risk === 'medium' ? 'bg-yellow-200 text-yellow-800' :
                        'bg-red-200 text-red-800'
                      }`}>
                        {qualityResult.plagiarism_check.similarity_risk.toUpperCase()} RISK
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">{qualityResult.plagiarism_check.recommendation}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Vocabulary uniqueness: {(qualityResult.plagiarism_check.vocabulary_uniqueness * 100).toFixed(1)}%
                    </p>
                  </div>

                  <p className="text-xs text-gray-400 text-center">
                    Analyzed {qualityResult.word_count} words &middot; {qualityResult.text_length} characters
                  </p>
                </div>
              )}

              {!qualityResult && !qualityLoading && (
                <div className="text-center py-10 text-gray-300">
                  <svg className="w-16 h-16 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                  </svg>
                  <p className="text-gray-400">Click "Analyze Quality" to check writing</p>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      {/* ═══ FOOTER ═══ */}
      <footer className={`mt-8 py-6 border-t ${t.footerBorder} ${t.footerBg} transition-colors duration-300`}>
        <div className="max-w-6xl mx-auto px-4 text-center">
          <p className={`text-2xl font-bold ${t.text}`}>AuthentiWrite</p>
          <p className={`text-base ${t.textSecondary} mt-2`}>Federal University Oye-Ekiti &middot; Department of Computer Science</p>
          <p className={`text-sm font-semibold ${t.text} mt-1`}>Project by Owofola Olakunle (FTP/CSC/25/0121610)</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
