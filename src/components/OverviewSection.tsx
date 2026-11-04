import React, { useState, useEffect } from 'react';
import {
  Database,
  Scale,
  Sliders,
  GitCompare,
  FlaskConical,
  ArrowRight,
  Code2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface OverviewSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
  onNavigate: (section: string) => void;
  isDetailedView?: boolean;
}

export const OverviewSection: React.FC<OverviewSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
  onNavigate,
  isDetailedView = false,
}) => {
  const [showCodeDrawer, setShowCodeDrawer] = useState(isDetailedView);

  useEffect(() => {
    setShowCodeDrawer(isDetailedView);
  }, [isDetailedView]);

  const noPCA = pipeline.regression.withoutPCA;
  const activePCA = pipeline.regression.withPCAByK[selectedK];
  const r2Delta = activePCA.r2 - noPCA.r2;

  const workflowSteps = [
    {
      step: '1',
      title: 'Dataset',
      metric: `${pipeline.totalRows} mixes`,
      sub: '8 features → MPa',
      section: 'dataset',
      icon: Database,
      color: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
    },
    {
      step: '2',
      title: 'StandardScaler',
      metric: 'μ = 0, σ = 1',
      sub: `Fit on ${pipeline.trainCount} train rows`,
      section: 'normalization',
      icon: Scale,
      color: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
    },
    {
      step: '3',
      title: 'PCA Transform',
      metric: `k = ${selectedK} PCs`,
      sub: `${(activePCA.retainedVarianceRatio * 100).toFixed(1)}% variance`,
      section: 'pca',
      icon: Sliders,
      color: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
    },
    {
      step: '4',
      title: 'OLS Regression',
      metric: `R² = ${activePCA.r2.toFixed(3)}`,
      sub: `vs ${noPCA.r2.toFixed(3)} (No PCA)`,
      section: 'regression',
      icon: GitCompare,
      color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
    },
    {
      step: '5',
      title: 'Mix Predictor',
      metric: 'Live MPa',
      sub: 'What-If Lab',
      section: 'predictor',
      icon: FlaskConical,
      color: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Clean Hero Banner + 3 Bullet Takeaways */}
      <div className="rounded-2xl bg-gradient-to-br from-slate-900/95 via-[#0B1325] to-slate-900/90 border border-slate-800/90 p-6 sm:p-8 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/25 text-cyan-300 text-xs font-medium">
              <Sparkles className="w-3.5 h-3.5" />
              Project Topic: PCA on Normalized Concrete Ingredients
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Does PCA Improve Concrete Strength Prediction?
            </h1>
            <ul className="space-y-2 text-sm text-slate-300 pt-1">
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-white">6 Principal Components</strong> capture{' '}
                  <strong className="text-cyan-300 font-mono">97.31%</strong> of total ingredient
                  variance (crossing the 95% threshold while cutting dimensions by 25%).
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-white">PCA blends all 8 features</strong>—weighted
                  loadings prove every ingredient (~11.5% to 12.5% each) contributes to the
                  retained components.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>
                  <strong className="text-white">Regression Trade-Off:</strong> Baseline OLS
                  achieves <strong className="text-emerald-300 font-mono">R² = 0.6275</strong>{' '}
                  (RMSE 9.80 MPa) vs.{' '}
                  <strong className="text-amber-300 font-mono">R² = 0.5679</strong> (RMSE 10.55 MPa)
                  with 6 PCs because the dropped 2.69% variance still correlates with strength.
                </span>
              </li>
            </ul>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex flex-col sm:flex-row lg:flex-col gap-3 shrink-0">
            <button
              onClick={() => onNavigate('pca')}
              className="px-5 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all"
            >
              Launch PCA Simulator
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('predictor')}
              className="px-5 py-3 rounded-xl bg-slate-800/90 hover:bg-slate-700 text-slate-100 border border-slate-700 font-semibold text-sm flex items-center justify-center gap-2 transition-all"
            >
              Try Live Strength Predictor
              <FlaskConical className="w-4 h-4 text-cyan-400" />
            </button>
          </div>
        </div>
      </div>

      {/* 4 Large, Clean Scoreboard Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-5">
          <div className="text-xs font-medium text-slate-400">Dataset &amp; Split (Seed 42)</div>
          <div className="text-2xl font-bold font-mono text-white mt-2">
            {pipeline.totalRows.toLocaleString()}{' '}
            <span className="text-sm font-normal text-slate-400">rows</span>
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono">
            {pipeline.trainCount} Train • {pipeline.testCount} Test • 0 Missing
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-cyan-500/30 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-cyan-300">
              Retained Variance (k = {selectedK})
            </span>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300">
              {selectedK}/8 PCs
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-cyan-300 mt-2">
            {(activePCA.retainedVarianceRatio * 100).toFixed(2)}%
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono">
            Lost: {(activePCA.lostVarianceRatio * 100).toFixed(2)}% • Saved:{' '}
            {activePCA.dimensionalityReductionPercent.toFixed(1)}%
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-emerald-500/30 p-5">
          <div className="text-xs font-medium text-emerald-300">
            Model A: Without PCA (8 Features)
          </div>
          <div className="text-2xl font-bold font-mono text-white mt-2">
            R² = {noPCA.r2.toFixed(4)}
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono">
            RMSE: {noPCA.rmse.toFixed(2)} MPa • MAE: {noPCA.mae.toFixed(2)}
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-purple-500/30 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-purple-300">
              Model B: With PCA (k = {selectedK})
            </span>
            <span
              className={`text-[11px] font-mono px-1.5 py-0.5 rounded ${
                Math.abs(r2Delta) < 1e-6
                  ? 'bg-emerald-500/15 text-emerald-300'
                  : 'bg-amber-500/15 text-amber-300'
              }`}
            >
              {r2Delta >= 0 ? '+' : ''}
              {(r2Delta * 100).toFixed(2)}% R²
            </span>
          </div>
          <div className="text-2xl font-bold font-mono text-white mt-2">
            R² = {activePCA.r2.toFixed(4)}
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono">
            RMSE: {activePCA.rmse.toFixed(2)} MPa • MAE: {activePCA.mae.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Interactive 5-Step Clickable Pipeline Map */}
      <div className="rounded-2xl bg-slate-900/75 border border-slate-800/90 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-5">
          <div>
            <h2 className="text-base font-bold text-white">
              Interactive Notebook Pipeline (Click Any Stage to Explore)
            </h2>
            <p className="text-xs text-slate-400">
              Follow the exact steps from your uploaded Google Colab notebook
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Quick PC Selector:</span>
            {[4, 6, 7, 8].map((k) => (
              <button
                key={k}
                onClick={() => onSelectK(k)}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
                  selectedK === k
                    ? 'bg-cyan-500 text-slate-950'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                k={k}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {workflowSteps.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.step}
                onClick={() => onNavigate(item.section)}
                className="group text-left p-4 rounded-xl bg-slate-950/80 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/40 transition-all flex flex-col justify-between"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className={`p-2 rounded-lg border ${item.color}`}>
                    <Icon className="w-4 h-4" />
                  </span>
                  <span className="text-xs font-mono text-slate-500 group-hover:text-cyan-400 flex items-center gap-1">
                    Step {item.step}
                    <ArrowRight className="w-3 h-3" />
                  </span>
                </div>
                <div>
                  <div className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                    {item.title}
                  </div>
                  <div className="text-base font-mono font-semibold text-slate-200 mt-1">
                    {item.metric}
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">{item.sub}</div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Expandable Drawer for Notebook Code & Verification */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowCodeDrawer(!showCodeDrawer)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <Code2 className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-slate-200">
              Notebook Code &amp; Exact Numerical Verification (Concrete_Strength.ipynb)
            </span>
          </div>
          <span className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            {showCodeDrawer ? 'Hide Details' : 'Show Code & Proof'}
            {showCodeDrawer ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </span>
        </button>

        {showCodeDrawer && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto leading-relaxed">
              <div className="text-slate-500 mb-2"># Extracted from Concrete_Strength.ipynb</div>
              <div>X_train, X_test, y_train, y_test = train_test_split(</div>
              <div className="pl-4">X, y, test_size=0.2, random_state=42</div>
              <div>)</div>
              <div>scaler = StandardScaler()</div>
              <div>X_train_scaled = scaler.fit_transform(X_train)</div>
              <div>X_test_scaled = scaler.transform(X_test)</div>
              <div className="mt-2">pca_95 = PCA(n_components=6) # 97.31% variance</div>
              <div>X_train_pca = pca_95.fit_transform(X_train_scaled)</div>
              <div>X_test_pca = pca_95.transform(X_test_scaled)</div>
            </div>

            <div className="bg-slate-950 rounded-xl p-4 border border-slate-800 text-xs space-y-2">
              <div className="font-semibold text-slate-200">
                Verified Notebook Outputs vs. Live App:
              </div>
              <div className="grid grid-cols-3 gap-2 font-mono text-[11px] pt-1">
                <div className="text-slate-400">Metric</div>
                <div className="text-slate-400">Colab Notebook</div>
                <div className="text-cyan-400">Simulator Engine</div>

                <div className="text-slate-300">PC1 Variance</div>
                <div>0.28016170</div>
                <div className="text-emerald-400">
                  {pipeline.pca.explainedVarianceRatio[0].toFixed(8)}
                </div>

                <div className="text-slate-300">k=6 Cum. Variance</div>
                <div>97.3099%</div>
                <div className="text-emerald-400">
                  {(pipeline.pca.cumulativeVarianceRatio[5] * 100).toFixed(4)}%
                </div>

                <div className="text-slate-300">RMSE (No PCA)</div>
                <div>9.796708</div>
                <div className="text-emerald-400">{noPCA.rmse.toFixed(6)}</div>

                <div className="text-slate-300">R² (No PCA)</div>
                <div>0.627542</div>
                <div className="text-emerald-400">{noPCA.r2.toFixed(6)}</div>

                <div className="text-slate-300">RMSE (PCA k=6)</div>
                <div>10.551508</div>
                <div className="text-emerald-400">
                  {pipeline.regression.withPCAByK[6].rmse.toFixed(6)}
                </div>

                <div className="text-slate-300">R² (PCA k=6)</div>
                <div>0.567938</div>
                <div className="text-emerald-400">
                  {pipeline.regression.withPCAByK[6].r2.toFixed(6)}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
