import React, { useState, useMemo, useEffect } from 'react';
import {
  BarChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  ScatterChart,
  Scatter,
  ComposedChart,
} from 'recharts';
import {
  Sliders,
  TrendingUp,
  Compass,
  Layers,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Grid,
} from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface PCASimulatorSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
  isDetailedView?: boolean;
}

export const PCASimulatorSection: React.FC<PCASimulatorSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
  isDetailedView = false,
}) => {
  const [activeTab, setActiveTab] = useState<'variance' | 'scatter' | 'contributions'>('variance');
  const [showLoadingsDrawer, setShowLoadingsDrawer] = useState<boolean>(isDetailedView);

  useEffect(() => {
    setShowLoadingsDrawer(isDetailedView);
  }, [isDetailedView]);

  const currentKMetrics = pipeline.regression.withPCAByK[selectedK];
  const noPCAMetrics = pipeline.regression.withoutPCA;
  const retainedPct = currentKMetrics.retainedVarianceRatio * 100;
  const lostPct = currentKMetrics.lostVarianceRatio * 100;
  const dimRedPct = currentKMetrics.dimensionalityReductionPercent;

  const varianceChartData = useMemo(() => {
    return pipeline.pca.explainedVarianceRatio.map((ratio, idx) => {
      const pcNum = idx + 1;
      return {
        pc: `PC${pcNum}`,
        pcNum,
        individualPct: Number((ratio * 100).toFixed(2)),
        cumulativePct: Number((pipeline.pca.cumulativeVarianceRatio[idx] * 100).toFixed(2)),
        isRetained: pcNum <= selectedK,
      };
    });
  }, [pipeline.pca, selectedK]);

  const weightedContributions = useMemo(() => {
    const items = pipeline.pca.featureImportanceByK[selectedK] || [];
    return items.map((item) => ({
      key: String(item.featureIndex),
      name: item.shortName,
      fullName: item.name,
      score: Number(item.importance.toFixed(6)),
      pctOfRetained: Number(item.sharePercent.toFixed(2)),
    }));
  }, [pipeline.pca.featureImportanceByK, selectedK]);

  const scatterBuckets = useMemo(() => {
    const low: { pc1: number; pc2: number; strength: number }[] = [];
    const mid: { pc1: number; pc2: number; strength: number }[] = [];
    const high: { pc1: number; pc2: number; strength: number }[] = [];

    pipeline.pca.projectedSamples.forEach((pt) => {
      const item = {
        pc1: Number(pt.pcCoords[0].toFixed(3)),
        pc2: Number(pt.pcCoords[1].toFixed(3)),
        strength: Number(pt.actualStrength.toFixed(2)),
      };
      if (pt.actualStrength < 25) low.push(item);
      else if (pt.actualStrength <= 45) mid.push(item);
      else high.push(item);
    });
    return { low, mid, high };
  }, [pipeline.pca.projectedSamples]);

  const getLoadingCellStyle = (val: number, isRetainedRow: boolean) => {
    const abs = Math.min(1, Math.abs(val));
    if (!isRetainedRow) {
      return { backgroundColor: 'rgba(15, 23, 42, 0.5)', color: '#475569' };
    }
    if (val >= 0) {
      return {
        backgroundColor: `rgba(6, 182, 212, ${0.1 + abs * 0.65})`,
        color: abs > 0.35 ? '#ffffff' : '#cbd5e1',
      };
    }
    return {
      backgroundColor: `rgba(244, 63, 94, ${0.1 + abs * 0.65})`,
      color: abs > 0.35 ? '#ffffff' : '#cbd5e1',
    };
  };

  return (
    <div className="space-y-6">
      {/* Pinned Interactive PCA Controller Card */}
      <div className="rounded-2xl bg-gradient-to-br from-slate-900 via-[#0C1528] to-slate-900 border border-cyan-500/35 p-6 shadow-xl space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-md bg-cyan-500/15 text-cyan-300 text-xs font-mono font-semibold mb-1">
              <Sliders className="w-3.5 h-3.5" />
              Interactive Principal Component Slider
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white">
              Number of Principal Components:{' '}
              <span className="text-cyan-400 font-mono">{selectedK}</span> of 8
            </h2>
          </div>

          {/* Direct 1..8 Buttons */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((k) => (
              <button
                key={k}
                onClick={() => onSelectK(k)}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all ${
                  selectedK === k
                    ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20 scale-105'
                    : k === 6
                    ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/25'
                    : 'bg-slate-800/90 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {k} {k === 6 ? '(95%★)' : ''}
              </button>
            ))}
          </div>
        </div>

        {/* Range Slider */}
        <div className="space-y-2">
          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={selectedK}
            onChange={(e) => onSelectK(Number(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer h-2"
          />
          <div className="flex justify-between text-[11px] font-mono text-slate-400">
            <span>1 PC (28.02% var)</span>
            <span className="text-emerald-400 font-semibold">
              ★ 6 PCs = 97.31% (≥95% Notebook Target)
            </span>
            <span>8 PCs (100.00% var)</span>
          </div>
        </div>

        {/* 4 Dynamic Metric Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-2">
          <div className="rounded-xl bg-slate-950/80 border border-cyan-500/30 p-4">
            <div className="text-xs text-slate-400">Explained Variance Retained</div>
            <div className="text-2xl font-bold font-mono text-cyan-300 mt-1">
              {retainedPct.toFixed(2)}%
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              {retainedPct >= 95 ? 'Meets ≥95% threshold' : 'Below 95% threshold'}
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/80 border border-rose-500/30 p-4">
            <div className="text-xs text-slate-400">Information Lost</div>
            <div className="text-2xl font-bold font-mono text-rose-300 mt-1">
              {lostPct.toFixed(2)}%
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              Dropped: {8 - selectedK} component{8 - selectedK === 1 ? '' : 's'}
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/80 border border-purple-500/30 p-4">
            <div className="text-xs text-slate-400">Dimensionality Reduction</div>
            <div className="text-2xl font-bold font-mono text-purple-300 mt-1">
              {dimRedPct.toFixed(1)}%
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              8 features → {selectedK} PC{selectedK > 1 ? 's' : ''}
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/80 border border-emerald-500/30 p-4">
            <div className="text-xs text-slate-400">PCA Regression Test R²</div>
            <div className="text-2xl font-bold font-mono text-emerald-300 mt-1">
              {currentKMetrics.r2.toFixed(4)}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5 font-mono">
              RMSE: {currentKMetrics.rmse.toFixed(2)} MPa (vs {noPCAMetrics.rmse.toFixed(2)})
            </div>
          </div>
        </div>

        {/* Concise 3-Bullet Core Concept */}
        <ul className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs text-slate-300">
          <li className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <span>
              <strong className="text-white">No Features Deleted:</strong> PCA constructs each PC
              as a weighted sum of all 8 standardized concrete ingredients.
            </span>
          </li>
          <li className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong className="text-white">95% Rule at k = 6:</strong> Retaining PC1–PC6
              preserves <strong className="text-cyan-300 font-mono">97.31%</strong> of variance
              while dropping only PC7 (2.30%) and PC8 (0.39%).
            </span>
          </li>
          <li className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
            <span>
              <strong className="text-white">Balanced Ingredient Usage:</strong> At k = 6, all 8
              ingredients contribute nearly equally (~11.5% to 12.5% each).
            </span>
          </li>
        </ul>
      </div>

      {/* Sub-Tab Bar for Primary Charts */}
      <div className="inline-flex flex-wrap rounded-xl bg-slate-900 border border-slate-800 p-1 gap-1">
        <button
          onClick={() => setActiveTab('variance')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'variance'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          1. Variance &amp; 95% Curve
        </button>
        <button
          onClick={() => setActiveTab('scatter')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'scatter'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Compass className="w-4 h-4" />
          2. 2D PCA Projection (PC1 vs PC2)
        </button>
        <button
          onClick={() => setActiveTab('contributions')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'contributions'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Layers className="w-4 h-4" />
          3. Original Features vs. PCs
        </button>
      </div>

      {/* TAB 1: Explained Variance Bar Chart & Cumulative Curve */}
      {(activeTab === 'variance' || isDetailedView) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Individual Explained Variance Bar Chart */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">
                Explained Variance by Component (Scree Plot)
              </h3>
              <p className="text-xs text-slate-400">
                Cyan bars are retained (PC1–PC{selectedK}); slate bars are discarded.
              </p>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={varianceChartData}
                  margin={{ top: 10, right: 15, left: 0, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="pc" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                  <YAxis unit="%" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px',
                    }}
                  />
                  <Bar
                    dataKey="individualPct"
                    name="Explained Variance (%)"
                    radius={[6, 6, 0, 0]}
                  >
                    {varianceChartData.map((entry, idx) => (
                      <Cell
                        key={`cell-${idx}`}
                        fill={entry.isRetained ? '#06b6d4' : '#334155'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Cumulative Explained Variance Curve with 95% Reference Line */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
            <div>
              <h3 className="text-base font-bold text-white">
                Cumulative Explained Variance Curve (95% Threshold)
              </h3>
              <p className="text-xs text-slate-400">
                Crosses the 95% reference line at k = 6 components (97.31% retained).
              </p>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={varianceChartData}
                  margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="pc" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                  <YAxis domain={[0, 105]} unit="%" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px',
                    }}
                  />
                  <ReferenceLine
                    y={95}
                    stroke="#10b981"
                    strokeDasharray="5 5"
                    label={{
                      value: '95% Target',
                      fill: '#10b981',
                      fontSize: 11,
                      position: 'insideBottomRight',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="cumulativePct"
                    name="Cumulative Variance (%)"
                    stroke="#a855f7"
                    strokeWidth={3}
                    dot={{ r: 5, fill: '#a855f7' }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: 2D PCA Visualization (PC1 vs PC2 colored by strength) */}
      {(activeTab === 'scatter' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-bold text-white">
                PCA 2D Visualization: PC1 (28.02%) vs. PC2 (17.91%)
              </h3>
              <p className="text-xs text-slate-400">
                All {pipeline.totalRows.toLocaleString()} concrete mixes projected onto the first
                two principal components and colored by Compressive Strength (MPa)
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="flex items-center gap-1.5 text-amber-400">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" /> &lt;25 MPa
              </span>
              <span className="flex items-center gap-1.5 text-cyan-400">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" /> 25–45 MPa
              </span>
              <span className="flex items-center gap-1.5 text-purple-400">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-400 inline-block" /> &gt;45 MPa
              </span>
            </div>
          </div>

          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="pc1"
                  name="PC1"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  label={{
                    value: 'Principal Component 1 (PC1)',
                    position: 'insideBottom',
                    offset: -5,
                    fill: '#94a3b8',
                    fontSize: 11,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="pc2"
                  name="PC2"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  label={{
                    value: 'PC2',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#94a3b8',
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                />
                <Scatter
                  name="Low (<25 MPa)"
                  data={scatterBuckets.low}
                  fill="#f59e0b"
                  opacity={0.75}
                />
                <Scatter
                  name="Medium (25-45 MPa)"
                  data={scatterBuckets.mid}
                  fill="#06b6d4"
                  opacity={0.75}
                />
                <Scatter
                  name="High (>45 MPa)"
                  data={scatterBuckets.high}
                  fill="#a855f7"
                  opacity={0.85}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* TAB 3: Original Features vs Principal Components (Weighted Contribution Bar Chart) */}
      {(activeTab === 'contributions' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              Original Features vs. Principal Components (k = {selectedK})
            </h3>
            <p className="text-xs text-slate-400">
              Weighted Feature Contribution ={' '}
              <code className="text-cyan-300">(loadings²) @ explained_variance_ratio</code>. Notice
              how all 8 ingredients actively contribute rather than any column being deleted.
            </p>
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={weightedContributions}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 30, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={95}
                  tick={{ fill: '#e2e8f0', fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                />
                <Bar
                  dataKey="score"
                  name="Weighted Variance Contribution"
                  fill="#06b6d4"
                  radius={[0, 6, 6, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Expandable Drawer for 8x8 PCA Loading Heatmap */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowLoadingsDrawer(!showLoadingsDrawer)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <Grid className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-slate-200">
              PCA Loading Matrix Heatmap (8 Principal Components × 8 Concrete Ingredients)
            </span>
          </div>
          <span className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            {showLoadingsDrawer ? 'Hide Heatmap' : 'Show Loading Heatmap'}
            {showLoadingsDrawer ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </span>
        </button>

        {showLoadingsDrawer && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 space-y-3 overflow-x-auto">
            <p className="text-xs text-slate-400">
              Rows PC1–PC{selectedK} are highlighted as active. Dimmed rows (PC{selectedK + 1}–PC8)
              are discarded at k = {selectedK}.
            </p>
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="p-2 text-left">Component</th>
                  <th className="p-2 text-right">Var %</th>
                  {pipeline.featureMeta.map((f) => (
                    <th key={f.key} className="p-2 text-center">
                      {f.shortName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pipeline.pca.components.map((row, rIdx) => {
                  const isRetained = rIdx + 1 <= selectedK;
                  return (
                    <tr
                      key={`pc-${rIdx}`}
                      className={`border-b border-slate-800/50 ${
                        !isRetained ? 'opacity-45' : ''
                      }`}
                    >
                      <td className="p-2 font-bold text-slate-200">
                        PC{rIdx + 1} {isRetained ? '✓' : ''}
                      </td>
                      <td className="p-2 text-right text-cyan-300">
                        {(pipeline.pca.explainedVarianceRatio[rIdx] * 100).toFixed(2)}%
                      </td>
                      {row.map((val, cIdx) => (
                        <td
                          key={`${rIdx}-${cIdx}`}
                          style={getLoadingCellStyle(val, isRetained)}
                          className="p-2 text-center border border-slate-900/60"
                        >
                          {val >= 0 ? '+' : ''}
                          {val.toFixed(3)}
                        </td>
                      ))}
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
