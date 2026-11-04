import React, { useMemo, useState, useEffect } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  LineChart,
  Line,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import {
  GitCompare,
  TrendingUp,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  BarChart3,
  Table,
} from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface RegressionComparisonSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
  isDetailedView?: boolean;
}

export const RegressionComparisonSection: React.FC<RegressionComparisonSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
  isDetailedView = false,
}) => {
  const [activeTab, setActiveTab] = useState<'curves' | 'scatter' | 'errors'>('curves');
  const [showImpactDrawer, setShowImpactDrawer] = useState<boolean>(isDetailedView);

  useEffect(() => {
    setShowImpactDrawer(isDetailedView);
  }, [isDetailedView]);

  const modelA = pipeline.regression.withoutPCA;
  const modelB = pipeline.regression.withPCAByK[selectedK];

  const kSeriesData = useMemo(() => {
    return [1, 2, 3, 4, 5, 6, 7, 8].map((k) => {
      const m = pipeline.regression.withPCAByK[k];
      return {
        kLabel: `k=${k}`,
        k,
        r2WithPCA: Number(m.r2.toFixed(4)),
        r2WithoutPCA: Number(modelA.r2.toFixed(4)),
        rmseWithPCA: Number(m.rmse.toFixed(3)),
        rmseWithoutPCA: Number(modelA.rmse.toFixed(3)),
        retainedPct: Number((m.retainedVarianceRatio * 100).toFixed(2)),
        lostPct: Number((m.lostVarianceRatio * 100).toFixed(2)),
        mae: Number(m.mae.toFixed(3)),
        mse: Number(m.mse.toFixed(2)),
      };
    });
  }, [pipeline.regression, modelA]);

  const scatterDataA = useMemo(
    () =>
      pipeline.regression.testActuals.map((actual, idx) => ({
        actual: Number(actual.toFixed(2)),
        predicted: Number((modelA.testPredictions[idx] ?? 0).toFixed(2)),
      })),
    [pipeline.regression.testActuals, modelA.testPredictions]
  );

  const scatterDataB = useMemo(
    () =>
      pipeline.regression.testActuals.map((actual, idx) => ({
        actual: Number(actual.toFixed(2)),
        predicted: Number((modelB.testPredictions[idx] ?? 0).toFixed(2)),
      })),
    [pipeline.regression.testActuals, modelB.testPredictions]
  );

  const errorBarData = [
    {
      metric: 'MAE (MPa)',
      WithoutPCA: Number(modelA.mae.toFixed(3)),
      WithPCA: Number(modelB.mae.toFixed(3)),
    },
    {
      metric: 'RMSE (MPa)',
      WithoutPCA: Number(modelA.rmse.toFixed(3)),
      WithPCA: Number(modelB.rmse.toFixed(3)),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top 3-Bullet Takeaway + Quick Component Selector */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800/90 p-6 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-2">
            <h2 className="text-xl font-bold text-white">
              Regression Without PCA vs. With PCA (OLS Linear Regression)
            </h2>
            <ul className="space-y-1.5 text-xs sm:text-sm text-slate-300">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  <strong className="text-white">Model A (Without PCA, 8 features):</strong>{' '}
                  Achieves <strong className="text-emerald-300 font-mono">R² = 0.6275</strong> and{' '}
                  <strong className="text-emerald-300 font-mono">RMSE = 9.80 MPa</strong>.
                </span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                <span>
                  <strong className="text-white">Model B (With PCA at k = 6):</strong> Retains{' '}
                  <strong className="text-cyan-300 font-mono">97.31%</strong> variance with{' '}
                  <strong className="text-amber-300 font-mono">R² = 0.5679</strong> (RMSE = 10.55
                  MPa).
                </span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0" />
                <span>
                  <strong className="text-white">Why Performance Drops at k = 6:</strong> PCA is
                  unsupervised—PC7 holds only 2.30% of X variance, yet carries crucial signal for
                  predicting strength (adding PC7 jumps R² to{' '}
                  <strong className="font-mono text-white">0.6255</strong>).
                </span>
              </li>
            </ul>
          </div>

          <div className="flex flex-col items-start lg:items-end gap-2 shrink-0">
            <span className="text-xs text-slate-400 font-medium">
              Select PCA Component Count (k):
            </span>
            <div className="flex items-center gap-1.5 flex-wrap">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((k) => (
                <button
                  key={k}
                  onClick={() => onSelectK(k)}
                  className={`w-8 h-8 rounded-lg text-xs font-mono font-bold transition-all ${
                    selectedK === k
                      ? 'bg-cyan-500 text-slate-950 shadow-sm'
                      : k === 6
                      ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Side-by-Side Model A vs Model B Scoreboard */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3">
          {/* Model A Card */}
          <div className="rounded-xl bg-slate-950/80 border border-emerald-500/35 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
                Model A: Regression Without PCA
              </span>
              <span className="text-xs font-mono text-slate-400">All 8 Scaled Features</span>
            </div>
            <div className="grid grid-cols-4 gap-2 pt-1 font-mono">
              <div>
                <div className="text-[11px] text-slate-500">R² Score</div>
                <div className="text-lg font-bold text-emerald-300">{modelA.r2.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">RMSE</div>
                <div className="text-lg font-bold text-white">{modelA.rmse.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MAE</div>
                <div className="text-lg font-bold text-slate-200">{modelA.mae.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MSE</div>
                <div className="text-lg font-bold text-slate-300">{modelA.mse.toFixed(1)}</div>
              </div>
            </div>
          </div>

          {/* Model B Card */}
          <div className="rounded-xl bg-slate-950/80 border border-cyan-500/35 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Model B: Regression With PCA (k = {selectedK})
              </span>
              <span className="text-xs font-mono text-cyan-300">
                {(modelB.retainedVarianceRatio * 100).toFixed(2)}% Variance
              </span>
            </div>
            <div className="grid grid-cols-4 gap-2 pt-1 font-mono">
              <div>
                <div className="text-[11px] text-slate-500">R² Score</div>
                <div className="text-lg font-bold text-cyan-300">{modelB.r2.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">RMSE</div>
                <div className="text-lg font-bold text-white">{modelB.rmse.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MAE</div>
                <div className="text-lg font-bold text-slate-200">{modelB.mae.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MSE</div>
                <div className="text-lg font-bold text-slate-300">{modelB.mse.toFixed(1)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sub-Tabs for Primary Visualizations */}
      <div className="inline-flex flex-wrap rounded-xl bg-slate-900 border border-slate-800 p-1 gap-1">
        <button
          onClick={() => setActiveTab('curves')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'curves'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          1. R² &amp; RMSE vs. Component Count
        </button>
        <button
          onClick={() => setActiveTab('scatter')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'scatter'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <GitCompare className="w-4 h-4" />
          2. Actual vs. Predicted Scatterplot
        </button>
        <button
          onClick={() => setActiveTab('errors')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'errors'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          3. Error Comparison (MAE &amp; RMSE)
        </button>
      </div>

      {/* TAB 1: R² & RMSE vs Number of Principal Components */}
      {(activeTab === 'curves' || isDetailedView) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* R2 vs PCs */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">
                R² Score vs. Number of Principal Components
              </h3>
              <p className="text-xs text-slate-400">
                Green dashed line = Baseline Without PCA (0.6275). Notice the jump from k=6 to k=7.
              </p>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={kSeriesData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="kLabel" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                  <YAxis domain={[0, 0.75]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px',
                    }}
                  />
                  <ReferenceLine
                    y={Number(modelA.r2.toFixed(4))}
                    stroke="#10b981"
                    strokeDasharray="5 5"
                    label={{
                      value: 'No PCA (0.6275)',
                      fill: '#10b981',
                      fontSize: 11,
                      position: 'insideBottomRight',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="r2WithPCA"
                    name="PCA R² Score"
                    stroke="#06b6d4"
                    strokeWidth={3}
                    dot={{ r: 5, fill: '#06b6d4' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* RMSE vs PCs */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">
                RMSE (MPa) vs. Number of Principal Components
              </h3>
              <p className="text-xs text-slate-400">
                Lower RMSE is better. At k=8, PCA RMSE (9.80 MPa) matches Without PCA identically.
              </p>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={kSeriesData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="kLabel" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                  <YAxis domain={[9, 17]} unit=" MPa" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px',
                    }}
                  />
                  <ReferenceLine
                    y={Number(modelA.rmse.toFixed(3))}
                    stroke="#10b981"
                    strokeDasharray="5 5"
                    label={{
                      value: 'No PCA (9.80 MPa)',
                      fill: '#10b981',
                      fontSize: 11,
                      position: 'insideTopRight',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="rmseWithPCA"
                    name="PCA RMSE (MPa)"
                    stroke="#f43f5e"
                    strokeWidth={3}
                    dot={{ r: 5, fill: '#f43f5e' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Actual vs Predicted Scatterplot */}
      {(activeTab === 'scatter' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              Actual vs. Predicted Concrete Strength ({pipeline.testCount} Test Samples)
            </h3>
            <p className="text-xs text-slate-400">
              Comparing Model A (Without PCA) and Model B (With PCA, k = {selectedK}) against the
              ideal y = x diagonal
            </p>
          </div>

          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="actual"
                  name="Actual Strength"
                  domain={[0, 85]}
                  unit=" MPa"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="predicted"
                  name="Predicted Strength"
                  domain={[0, 85]}
                  unit=" MPa"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                />
                <Legend />
                <ReferenceLine
                  segment={[
                    { x: 0, y: 0 },
                    { x: 85, y: 85 },
                  ]}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                />
                <Scatter
                  name="Without PCA (8 features)"
                  data={scatterDataA}
                  fill="#10b981"
                  opacity={0.7}
                />
                <Scatter
                  name={`With PCA (k=${selectedK})`}
                  data={scatterDataB}
                  fill="#06b6d4"
                  opacity={0.75}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* TAB 3: Error Comparison Chart */}
      {(activeTab === 'errors' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              Prediction Error Comparison: MAE &amp; RMSE (MPa)
            </h3>
            <p className="text-xs text-slate-400">
              Lower bars indicate higher accuracy on the {pipeline.testCount} unseen test samples
            </p>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={errorBarData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <YAxis unit=" MPa" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                />
                <Legend />
                <Bar
                  dataKey="WithoutPCA"
                  name="Without PCA (8 Features)"
                  fill="#10b981"
                  radius={[6, 6, 0, 0]}
                />
                <Bar
                  dataKey="WithPCA"
                  name={`With PCA (k=${selectedK})`}
                  fill="#06b6d4"
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Expandable Drawer: Full 1-to-8 PCA Impact Analysis Table */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowImpactDrawer(!showImpactDrawer)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <Table className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-slate-200">
              Complete PCA Impact Analysis Table (All Component Counts k = 1 to 8)
            </span>
          </div>
          <span className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            {showImpactDrawer ? 'Hide Table' : 'Show 1–8 Impact Table'}
            {showImpactDrawer ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </span>
        </button>

        {showImpactDrawer && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-2 text-left">PCs (k)</th>
                  <th className="py-2.5 px-2 text-right">Retained Var %</th>
                  <th className="py-2.5 px-2 text-right">Info Lost %</th>
                  <th className="py-2.5 px-2 text-right">MAE (MPa)</th>
                  <th className="py-2.5 px-2 text-right">RMSE (MPa)</th>
                  <th className="py-2.5 px-2 text-right">R² Score</th>
                  <th className="py-2.5 px-2 text-left pl-4 font-sans">Practical Verdict</th>
                </tr>
              </thead>
              <tbody>
                {kSeriesData.map((row) => {
                  const isSelected = row.k === selectedK;
                  return (
                    <tr
                      key={row.k}
                      onClick={() => onSelectK(row.k)}
                      className={`border-b border-slate-800/50 cursor-pointer transition-colors ${
                        isSelected ? 'bg-cyan-500/15' : 'hover:bg-slate-800/40'
                      }`}
                    >
                      <td className="py-2 px-2 font-bold text-white">
                        k = {row.k} {row.k === 6 ? '★' : ''}
                      </td>
                      <td className="py-2 px-2 text-right text-cyan-300">{row.retainedPct}%</td>
                      <td className="py-2 px-2 text-right text-rose-300">{row.lostPct}%</td>
                      <td className="py-2 px-2 text-right text-slate-300">{row.mae}</td>
                      <td className="py-2 px-2 text-right text-white font-semibold">
                        {row.rmseWithPCA}
                      </td>
                      <td className="py-2 px-2 text-right text-emerald-300 font-bold">
                        {row.r2WithPCA}
                      </td>
                      <td className="py-2 px-2 pl-4 font-sans text-slate-300">
                        {row.k === 6
                          ? 'Notebook ≥95% variance threshold (25% dim. reduction)'
                          : row.k === 7
                          ? 'Best balance: 99.61% variance & 0.6255 R²'
                          : row.k === 8
                          ? 'Zero loss — identical to Regression Without PCA'
                          : 'Underfits target strength'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
