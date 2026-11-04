import React, { useState, useMemo, useRef } from 'react';
import * as XLSX from 'xlsx';
import {
  Layers,
  Database,
  Scale,
  Sliders,
  GitCompare,
  FlaskConical,
  Lightbulb,
  Upload,
  Presentation,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  CheckCircle2,
  AlertCircle,
  FileSpreadsheet,
  Eye,
} from 'lucide-react';
import { runConcreteMLPipeline } from './ml/mlEngine';
import { OverviewSection } from './components/OverviewSection';
import { DatasetExplorerSection } from './components/DatasetExplorerSection';
import { NormalizationSection } from './components/NormalizationSection';
import { PCASimulatorSection } from './components/PCASimulatorSection';
import { RegressionComparisonSection } from './components/RegressionComparisonSection';
import { PredictorAndLabSection } from './components/PredictorAndLabSection';
import { ModelInsightsSection } from './components/ModelInsightsSection';

type SectionId =
  | 'overview'
  | 'dataset'
  | 'normalization'
  | 'pca'
  | 'regression'
  | 'predictor'
  | 'insights';

interface NavSectionItem {
  id: SectionId;
  label: string;
  badge?: string;
  icon: React.ComponentType<{ className?: string }>;
  presentationSummary: string;
}

const NAV_SECTIONS: NavSectionItem[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: Layers,
    presentationSummary:
      'Introduce the 6-stage ML pipeline from Concrete_Strength.ipynb: 1,030 concrete mixes, 80/20 train-test split (random_state=42), StandardScaler, PCA (95% variance threshold at k=6), and OLS Linear Regression comparison.',
  },
  {
    id: 'dataset',
    label: 'Dataset Explorer',
    badge: '1,030',
    icon: Database,
    presentationSummary:
      'Show the 8 concrete ingredients and target Compressive Strength (MPa). Highlight strong positive correlation with Cement (+0.50) and negative correlation with Water (-0.29).',
  },
  {
    id: 'normalization',
    label: 'Normalization',
    icon: Scale,
    presentationSummary:
      'Explain why StandardScaler (z = (x - μ_train) / σ_train) is mandatory before PCA: Coarse Aggregate (~973 kg/m³) has >2,500x larger raw variance than Superplasticizer (~6.2 kg/m³).',
  },
  {
    id: 'pca',
    label: 'PCA Simulator',
    badge: 'Core',
    icon: Sliders,
    presentationSummary:
      'Demonstrate the interactive k = 1..8 slider. Show that k = 6 crosses the 95% cumulative variance threshold (97.31% retained) and that PCA combines all 8 features rather than deleting columns.',
  },
  {
    id: 'regression',
    label: 'Regression Comparison',
    icon: GitCompare,
    presentationSummary:
      'Compare Model A (Without PCA: R² = 0.6275, RMSE = 9.80 MPa) vs Model B (With PCA at k = 6: R² = 0.5679, RMSE = 10.55 MPa). Explain why dropping PC7–PC8 trades 5.96% R² for 25% dimensionality reduction.',
  },
  {
    id: 'predictor',
    label: 'Concrete Strength Predictor',
    icon: FlaskConical,
    presentationSummary:
      'Run live concrete mix predictions and What-If experiments. Adjust Cement, Water, or Age to show how both trained models respond in real time.',
  },
  {
    id: 'insights',
    label: 'Model Insights',
    icon: Lightbulb,
    presentationSummary:
      'Summarize project conclusions, 3-point presentation summary, and Professor Viva Q&A cards.',
  },
];

export function App() {
  const [activeSection, setActiveSection] = useState<SectionId>('overview');
  const [selectedK, setSelectedK] = useState<number>(6);
  const [viewMode, setViewMode] = useState<'simple' | 'detailed'>('simple');
  const [presentationMode, setPresentationMode] = useState<boolean>(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  // Custom dataset state (defaults to built-in 1,030-row Concrete_Data.xls)
  const [customRows, setCustomRows] = useState<number[][] | undefined>(undefined);
  const [customDatasetName, setCustomDatasetName] = useState<string | null>(null);
  const [uploadModalOpen, setUploadModalOpen] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const pipeline = useMemo(() => {
    return runConcreteMLPipeline(
      customRows,
      customDatasetName || 'Concrete_Data.xls (Notebook Dataset)'
    );
  }, [customRows, customDatasetName]);

  const activeNavIndex = NAV_SECTIONS.findIndex((s) => s.id === activeSection);
  const activeNavMeta = NAV_SECTIONS[activeNavIndex] || NAV_SECTIONS[0];
  const currentKMetrics = pipeline.regression.withPCAByK[selectedK];
  const isDetailedView = viewMode === 'detailed';

  const handleNavigate = (section: string) => {
    if (NAV_SECTIONS.some((s) => s.id === section)) {
      setActiveSection(section as SectionId);
      setMobileMenuOpen(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const arrayBuffer = await file.arrayBuffer();
      const workbook = XLSX.read(arrayBuffer, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      if (!firstSheetName) {
        setUploadError('The uploaded file contains no sheets.');
        return;
      }

      const worksheet = workbook.Sheets[firstSheetName];
      const rawJson = XLSX.utils.sheet_to_json<unknown[]>(worksheet, { header: 1 });

      const numericRows: number[][] = [];
      for (let i = 0; i < rawJson.length; i++) {
        const row = rawJson[i];
        if (!Array.isArray(row) || row.length < 9) continue;
        const parsed = row.slice(0, 9).map((val) => Number(val));
        if (parsed.every((v) => Number.isFinite(v))) {
          numericRows.push(parsed);
        }
      }

      if (numericRows.length < 30) {
        setUploadError(
          `Found only ${numericRows.length} valid 9-column numeric rows. Please upload a CSV or Excel dataset with at least 30 rows and 9 numeric columns.`
        );
        return;
      }

      setCustomRows(numericRows);
      setCustomDatasetName(file.name);
      setUploadSuccess(
        `Loaded "${file.name}" (${numericRows.length.toLocaleString()} rows × 9 columns) and recomputed StandardScaler, PCA, and OLS Regression.`
      );
    } catch (err) {
      setUploadError(
        err instanceof Error
          ? `Failed to parse file: ${err.message}`
          : 'Failed to parse the uploaded dataset file.'
      );
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleResetDataset = () => {
    setCustomRows(undefined);
    setCustomDatasetName(null);
    setUploadError(null);
    setUploadSuccess(
      'Restored original Concrete_Data.xls dataset (1,030 records) from Concrete_Strength.ipynb.'
    );
  };

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 flex flex-col lg:flex-row selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Sidebar Navigation (Desktop) */}
      <aside className="hidden lg:flex lg:flex-col lg:w-72 lg:fixed lg:inset-y-0 bg-[#0B1120]/95 border-r border-slate-800/80 backdrop-blur-xl z-30">
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 via-cyan-400 to-purple-600 p-[1px] shadow-lg shadow-cyan-500/15 shrink-0">
              <div className="w-full h-full bg-[#090E1A] rounded-[11px] flex items-center justify-center">
                <FlaskConical className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white leading-snug">
                Concrete Strength AI Lab
              </h1>
              <p className="text-[11px] text-slate-400 leading-tight mt-0.5">
                Interactive PCA, Variance Analysis &amp; Regression Prediction Simulator
              </p>
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Simulator Modules
          </div>
          {NAV_SECTIONS.map((item, idx) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigate(item.id)}
                className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-left text-xs font-medium transition-all duration-150 group ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/15 via-blue-500/10 to-purple-500/10 text-cyan-300 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <span
                    className={`w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-mono shrink-0 transition-colors ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'bg-slate-900/80 text-slate-500 group-hover:text-slate-300'
                    }`}
                  >
                    {idx + 1}
                  </span>
                  <Icon
                    className={`w-4 h-4 shrink-0 ${
                      isActive ? 'text-cyan-400' : 'text-slate-400 group-hover:text-slate-300'
                    }`}
                  />
                  <span className="truncate">{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${
                      isActive
                        ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                        : 'bg-slate-900 border-slate-800 text-slate-400'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Global Persistent PCA Quick Controller in Sidebar */}
        <div className="p-4 m-3 rounded-xl bg-slate-900/90 border border-slate-800/90 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              Global PCA State
            </span>
            <span className="text-xs font-mono font-bold text-cyan-300 bg-cyan-500/15 border border-cyan-500/30 px-2 py-0.5 rounded">
              k = {selectedK} / 8
            </span>
          </div>

          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={selectedK}
            onChange={(e) => setSelectedK(Number(e.target.value))}
            aria-label="Global Number of Principal Components"
            className="w-full accent-cyan-400 cursor-pointer"
          />

          <div className="grid grid-cols-2 gap-2 pt-1 text-[11px]">
            <div className="bg-slate-950/70 rounded-lg p-2 border border-slate-800/80">
              <div className="text-slate-500 text-[10px]">Retained Var.</div>
              <div className="font-mono font-semibold text-emerald-400 mt-0.5">
                {(currentKMetrics.retainedVarianceRatio * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-slate-950/70 rounded-lg p-2 border border-slate-800/80">
              <div className="text-slate-500 text-[10px]">PCA Test R²</div>
              <div className="font-mono font-semibold text-cyan-300 mt-0.5">
                {currentKMetrics.r2.toFixed(4)}
              </div>
            </div>
          </div>

          {/* Dataset Status & Upload Button */}
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="text-[10px] text-slate-500 truncate">
                {customDatasetName ? 'Custom Dataset' : 'Notebook Dataset'}
              </div>
              <div className="text-[11px] font-mono text-slate-300 truncate">
                {customDatasetName || 'Concrete_Data.xls'} ({pipeline.totalRows})
              </div>
            </div>
            <button
              onClick={() => {
                setUploadError(null);
                setUploadSuccess(null);
                setUploadModalOpen(true);
              }}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-[11px] font-medium flex items-center gap-1 shrink-0 transition-colors"
            >
              <Upload className="w-3 h-3 text-cyan-400" />
              Data
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-72 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="sticky top-0 z-20 bg-[#070B14]/85 backdrop-blur-xl border-b border-slate-800/80 px-4 sm:px-6 lg:px-8 py-3.5">
          <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
            {/* Mobile Menu Button & Active Section Title */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white"
                aria-label="Toggle navigation menu"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                    Step {activeNavIndex + 1} of {NAV_SECTIONS.length}
                  </span>
                  <h2 className="text-base sm:text-lg font-bold text-white tracking-tight">
                    {activeNavMeta.label}
                  </h2>
                </div>
              </div>
            </div>

            {/* Right Header Controls */}
            <div className="flex items-center gap-2.5 flex-wrap">
              {/* Simple View vs Detailed View Mode Switcher */}
              <div className="inline-flex items-center rounded-xl bg-slate-900 border border-slate-800 p-1">
                <button
                  onClick={() => setViewMode('simple')}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                    viewMode === 'simple'
                      ? 'bg-cyan-500 text-slate-950 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Eye className="w-3.5 h-3.5" />
                  Simple View
                </button>
                <button
                  onClick={() => setViewMode('detailed')}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                    viewMode === 'detailed'
                      ? 'bg-purple-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Full Details
                </button>
              </div>

              {/* Presentation Speaker Notes Toggle */}
              <button
                onClick={() => setPresentationMode(!presentationMode)}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium border flex items-center gap-1.5 transition-all ${
                  presentationMode
                    ? 'bg-purple-500/20 border-purple-500/50 text-purple-200 shadow-sm'
                    : 'bg-slate-900/90 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <Presentation className="w-3.5 h-3.5 text-purple-400" />
                <span className="hidden sm:inline">Speaker Notes</span>
              </button>
            </div>
          </div>

          {/* Mobile Drawer Navigation */}
          {mobileMenuOpen && (
            <div className="lg:hidden mt-3 pt-3 border-t border-slate-800 grid grid-cols-1 sm:grid-cols-2 gap-1.5">
              {NAV_SECTIONS.map((item, idx) => {
                const Icon = item.icon;
                const isActive = activeSection === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleNavigate(item.id)}
                    className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium text-left ${
                      isActive
                        ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                        : 'text-slate-400 hover:bg-slate-900'
                    }`}
                  >
                    <span className="font-mono text-[10px] text-slate-500">{idx + 1}.</span>
                    <Icon className="w-4 h-4 text-cyan-400" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          )}
        </header>

        {/* Guided Presentation / Viva Banner */}
        {presentationMode && (
          <div className="bg-gradient-to-r from-purple-950/60 via-slate-900/90 to-cyan-950/60 border-b border-purple-500/30 px-4 sm:px-6 lg:px-8 py-3">
            <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex items-start gap-3">
                <div className="px-2 py-1 rounded bg-purple-500/20 border border-purple-500/40 text-purple-300 text-[11px] font-mono font-semibold shrink-0 mt-0.5">
                  Speaker Note • {activeNavIndex + 1}/{NAV_SECTIONS.length}
                </div>
                <p className="text-xs sm:text-sm text-slate-200 leading-relaxed">
                  {activeNavMeta.presentationSummary}
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                <button
                  onClick={() =>
                    handleNavigate(
                      NAV_SECTIONS[
                        (activeNavIndex - 1 + NAV_SECTIONS.length) % NAV_SECTIONS.length
                      ].id
                    )
                  }
                  className="px-2.5 py-1 rounded-lg bg-slate-900/90 hover:bg-slate-800 border border-slate-700 text-xs text-slate-300 flex items-center gap-1"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  Prev
                </button>
                <button
                  onClick={() =>
                    handleNavigate(NAV_SECTIONS[(activeNavIndex + 1) % NAV_SECTIONS.length].id)
                  }
                  className="px-2.5 py-1 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-xs text-cyan-200 font-medium flex items-center gap-1"
                >
                  Next
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Active Module Content */}
        <main className="flex-1 px-4 sm:px-6 lg:px-8 py-6 max-w-7xl w-full mx-auto">
          {activeSection === 'overview' && (
            <OverviewSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
              onNavigate={handleNavigate}
              isDetailedView={isDetailedView}
            />
          )}

          {activeSection === 'dataset' && (
            <DatasetExplorerSection
              pipeline={pipeline}
              onOpenUploadModal={() => {
                setUploadError(null);
                setUploadSuccess(null);
                setUploadModalOpen(true);
              }}
              isDetailedView={isDetailedView}
            />
          )}

          {activeSection === 'normalization' && (
            <NormalizationSection pipeline={pipeline} isDetailedView={isDetailedView} />
          )}

          {activeSection === 'pca' && (
            <PCASimulatorSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
              isDetailedView={isDetailedView}
            />
          )}

          {activeSection === 'regression' && (
            <RegressionComparisonSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
              isDetailedView={isDetailedView}
            />
          )}

          {activeSection === 'predictor' && (
            <PredictorAndLabSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
              isDetailedView={isDetailedView}
            />
          )}

          {activeSection === 'insights' && (
            <ModelInsightsSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
              isDetailedView={isDetailedView}
            />
          )}
        </main>

        {/* Bottom Step Navigation Footer */}
        <footer className="border-t border-slate-800/80 px-4 sm:px-6 lg:px-8 py-4 mt-8 bg-[#060911]">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
            <div>
              Concrete Strength AI Lab •{' '}
              <span className="font-mono text-slate-300">Concrete_Strength.ipynb</span> (
              {pipeline.totalRows} mixes)
            </div>
            <div className="flex items-center gap-2">
              {activeNavIndex > 0 && (
                <button
                  onClick={() => handleNavigate(NAV_SECTIONS[activeNavIndex - 1].id)}
                  className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 flex items-center gap-1"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  {NAV_SECTIONS[activeNavIndex - 1].label}
                </button>
              )}
              {activeNavIndex < NAV_SECTIONS.length - 1 && (
                <button
                  onClick={() => handleNavigate(NAV_SECTIONS[activeNavIndex + 1].id)}
                  className="px-3 py-1.5 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-300 font-medium flex items-center gap-1"
                >
                  Next: {NAV_SECTIONS[activeNavIndex + 1].label}
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        </footer>
      </div>

      {/* Dataset Upload & Verification Modal */}
      {uploadModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2.5">
                <FileSpreadsheet className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-bold text-white">
                  Dataset Source &amp; Custom CSV/Excel Upload
                </h3>
              </div>
              <button
                onClick={() => setUploadModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800 text-xs space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Active Dataset:</span>
                <span className="font-mono font-semibold text-cyan-300">
                  {customDatasetName || 'Concrete_Data.xls (Notebook Default)'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Total Records × Columns:</span>
                <span className="font-mono text-slate-200">
                  {pipeline.totalRows.toLocaleString()} rows × 9 columns
                </span>
              </div>
            </div>

            {uploadError && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <span>{uploadError}</span>
              </div>
            )}

            {uploadSuccess && (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 flex items-start gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>{uploadSuccess}</span>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
              <div className="flex items-center gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xls,.xlsx"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="concrete-dataset-upload"
                />
                <label
                  htmlFor="concrete-dataset-upload"
                  className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs cursor-pointer flex items-center gap-1.5 transition-colors"
                >
                  <Upload className="w-3.5 h-3.5" />
                  Upload CSV / Excel
                </label>

                {customDatasetName && (
                  <button
                    onClick={handleResetDataset}
                    className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium flex items-center gap-1.5 transition-colors"
                  >
                    <RotateCcw className="w-3.5 h-3.5 text-amber-400" />
                    Restore Notebook Dataset
                  </button>
                )}
              </div>

              <button
                onClick={() => setUploadModalOpen(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
