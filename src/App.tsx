import React, { useState, useMemo } from 'react';
import { Database, Sliders, GitCompare, FlaskConical, Menu, X } from 'lucide-react';
import { runConcreteMLPipeline } from './ml/mlEngine';
import { DatasetExplorerSection } from './components/DatasetExplorerSection';
import { PCASimulatorSection } from './components/PCASimulatorSection';
import { RegressionComparisonSection } from './components/RegressionComparisonSection';
import { PredictorAndLabSection } from './components/PredictorAndLabSection';

type TabId = 'dataset' | 'pca' | 'regression' | 'predictor';

const TABS: {
  id: TabId;
  label: string;
  subtitle: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  {
    id: 'dataset',
    label: '1. Dataset & Scaling',
    subtitle: '1,030 mixes & StandardScaler',
    icon: Database,
  },
  {
    id: 'pca',
    label: '2. PCA Simulator',
    subtitle: 'Interactive 1–8 PC variance lab',
    icon: Sliders,
  },
  {
    id: 'regression',
    label: '3. Regression Comparison',
    subtitle: 'Without PCA vs. With PCA',
    icon: GitCompare,
  },
  {
    id: 'predictor',
    label: '4. Strength Predictor',
    subtitle: 'Live MPa concrete mix predictor',
    icon: FlaskConical,
  },
];

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>('pca');
  const [selectedK, setSelectedK] = useState<number>(6);
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);

  const pipeline = useMemo(() => runConcreteMLPipeline(), []);
  const currentTabMeta = TABS.find((t) => t.id === activeTab) || TABS[0];
  const currentKMetrics = pipeline.regression.withPCAByK[selectedK];

  const handleSelectTab = (id: TabId) => {
    setActiveTab(id);
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 flex flex-col lg:flex-row">
      {/* Left Sidebar (Desktop) */}
      <aside className="hidden lg:flex lg:flex-col lg:w-68 lg:fixed lg:inset-y-0 bg-[#0B1120] border-r border-slate-800/90 z-30">
        {/* Brand Title */}
        <div className="p-5 border-b border-slate-800/80 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-slate-950 font-bold shadow-md shadow-cyan-500/15 shrink-0">
            <FlaskConical className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight leading-snug">
              Concrete Strength AI Lab
            </h1>
            <p className="text-[11px] text-slate-400 leading-tight mt-0.5">
              PCA &amp; Regression Simulator
            </p>
          </div>
        </div>

        {/* 4 Vertical Sidebar Tabs */}
        <nav className="flex-1 p-3 space-y-2">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => handleSelectTab(tab.id)}
                className={`w-full p-3.5 rounded-xl text-left flex items-center gap-3 transition-all ${
                  isActive
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/15'
                    : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    isActive ? 'text-slate-950' : 'text-cyan-400'
                  }`}
                />
                <div className="min-w-0">
                  <div className="text-xs font-semibold truncate">{tab.label}</div>
                  <div
                    className={`text-[11px] truncate mt-0.5 ${
                      isActive ? 'text-slate-900/80 font-medium' : 'text-slate-500'
                    }`}
                  >
                    {tab.subtitle}
                  </div>
                </div>
              </button>
            );
          })}
        </nav>

        {/* Compact Global PCA Status at Bottom of Sidebar */}
        <div className="p-4 m-3 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 font-medium">Active Components</span>
            <span className="font-mono font-bold text-cyan-400">k = {selectedK} / 8</span>
          </div>
          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={selectedK}
            onChange={(e) => setSelectedK(Number(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer"
          />
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>
              Var: <strong className="text-emerald-400">{(currentKMetrics.retainedVarianceRatio * 100).toFixed(1)}%</strong>
            </span>
            <span>
              R²: <strong className="text-cyan-300">{currentKMetrics.r2.toFixed(3)}</strong>
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 lg:pl-68 flex flex-col min-w-0">
        {/* Clean Top Header */}
        <header className="sticky top-0 z-20 bg-[#070B14]/90 backdrop-blur-xl border-b border-slate-800/80 px-4 sm:px-6 lg:px-8 py-4">
          <div className="max-w-6xl mx-auto flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setMobileOpen(!mobileOpen)}
                className="lg:hidden p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-white"
                aria-label="Toggle menu"
              >
                {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              <div>
                <h2 className="text-base sm:text-lg font-bold text-white">
                  {currentTabMeta.label}
                </h2>
                <p className="text-xs text-slate-400">
                  Interactive PCA, Variance Analysis &amp; Regression Prediction Simulator
                </p>
              </div>
            </div>

            <div className="hidden sm:flex items-center gap-2 text-xs font-mono bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-xl">
              <span className="text-slate-400">Selected PCs:</span>
              <span className="text-cyan-400 font-bold">k = {selectedK}</span>
              <span className="text-slate-600">|</span>
              <span className="text-emerald-400">
                {(currentKMetrics.retainedVarianceRatio * 100).toFixed(2)}% Variance
              </span>
            </div>
          </div>

          {/* Mobile Menu Dropdown */}
          {mobileOpen && (
            <div className="lg:hidden mt-3 pt-3 border-t border-slate-800 space-y-1.5">
              {TABS.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => handleSelectTab(tab.id)}
                    className={`w-full px-3.5 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2.5 ${
                      isActive
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'text-slate-300 hover:bg-slate-900'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          )}
        </header>

        {/* Active Tab Content */}
        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {activeTab === 'dataset' && <DatasetExplorerSection pipeline={pipeline} />}

          {activeTab === 'pca' && (
            <PCASimulatorSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
            />
          )}

          {activeTab === 'regression' && (
            <RegressionComparisonSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
            />
          )}

          {activeTab === 'predictor' && (
            <PredictorAndLabSection
              pipeline={pipeline}
              selectedK={selectedK}
              onSelectK={setSelectedK}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
