import React, { useMemo } from 'react';
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
import { Sliders, CheckCircle2 } from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface PCASimulatorSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
}

export const PCASimulatorSection: React.FC<PCASimulatorSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
}) => {
  const currentKMetrics = pipeline.regression.withPCAByK[selectedK];
  const retainedPct = currentKMetrics.retainedVarianceRatio * 100;
  const lostPct = currentKMetrics.lostVarianceRatio * 100;
  const dimRedPct = currentKMetrics.dimensionalityReductionPercent;

  const varianceChartData = useMemo(() => {
    return pipeline.pca.explainedVarianceRatio.map((ratio, idx) => {
      const pcNum = idx + 1;
      return {
        pc: `PC${pcNum}`,
        individualPct: Number((ratio * 100).toFixed(2)),
        cumulativePct: Number((pipeline.pca.cumulativeVarianceRatio[idx] * 100).toFixed(2)),
        isRetained: pcNum <= selectedK,
      };
    });
  }, [pipeline.pca, selectedK]);

  const weightedContributions = useMemo(() => {
    const items = pipeline.pca.featureImportanceByK[selectedK] || [];
    return items.map((item) => ({
      name: item.shortName,
      score: Number(item.importance.toFixed(4)),
      sharePct: Number(item.sharePercent.toFixed(1)),
    }));
  }, [pipeline.pca.featureImportanceByK, selectedK]);

  const scatterBuckets = useMemo(() => {
    const low: { pc1: number; pc2: number; strength: number }[] = [];
    const mid: { pc1: number; pc2: number; strength: number }[] = [];
    const high: { pc1: number; pc2: number; strength: number }[] = [];

    pipeline.pca.projectedSamples.forEach((pt) => {
      const item = {
        pc1: Number(pt.pcCoords[0].toFixed(2)),
        pc2: Number(pt.pcCoords[1].toFixed(2)),
        strength: Number(pt.actualStrength.toFixed(1)),
      };
      if (pt.actualStrength < 25) low.push(item);
      else if (pt.actualStrength <= 45) mid.push(item);
      else high.push(item);
    });
    return { low, mid, high };
  }, [pipeline.pca.projectedSamples]);

  return (
    <div className="space-y-6">
      {/* Main Interactive Slider Card */}
      <div className="rounded-2xl bg-slate-900/90 border border-cyan-500/35 p-6 space-y-5 shadow-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <Sliders className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-xl font-bold text-white">
                Number of Principal Components:{' '}
                <span className="text-cyan-400 font-mono">{selectedK}</span> of 8
              </h2>
              <p className="text-xs text-slate-400">
                Move the slider to see how changing the number of components affects retained variance
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 flex-wrap">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((k) => (
              <button
                key={k}
                onClick={() => onSelectK(k)}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all ${
                  selectedK === k
                    ? 'bg-cyan-500 text-slate-950 shadow-sm'
                    : k === 6
                    ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/40'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {k} {k === 6 ? '(95%★)' : ''}
              </button>
            ))}
          </div>
        </div>

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
          <div className="flex justify-between text-xs font-mono text-slate-400">
            <span>1 PC (28.0% variance)</span>
            <span className="text-emerald-400 font-semibold">
              ★ 6 PCs = 97.31% Variance (Notebook ≥95% Target)
            </span>
            <span>8 PCs (100% variance)</span>
          </div>
        </div>

        {/* 3 Simple, High-Impact KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
          <div className="rounded-xl bg-slate-950/90 border border-cyan-500/30 p-4">
            <div className="text-xs text-slate-400">Total Explained Variance Retained</div>
            <div className="text-3xl font-bold font-mono text-cyan-300 mt-1">
              {retainedPct.toFixed(2)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {retainedPct >= 95 ? 'Passes 95% threshold' : 'Below 95% threshold'}
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/90 border border-rose-500/30 p-4">
            <div className="text-xs text-slate-400">Information Lost</div>
            <div className="text-3xl font-bold font-mono text-rose-300 mt-1">
              {lostPct.toFixed(2)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {8 - selectedK} component{8 - selectedK === 1 ? '' : 's'} dropped
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/90 border border-purple-500/30 p-4">
            <div className="text-xs text-slate-400">Dimensionality Reduction</div>
            <div className="text-3xl font-bold font-mono text-purple-300 mt-1">
              {dimRedPct.toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Compressed from 8 → {selectedK} features
            </div>
          </div>
        </div>
      </div>

      {/* Row 1: Explained Variance Bar Chart & Cumulative Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              1. Variance Explained by Each Component
            </h3>
            <p className="text-xs text-slate-400">
              Cyan bars are kept (PC1–PC{selectedK}); gray bars are dropped
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={varianceChartData} margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
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
                <Bar dataKey="individualPct" name="Variance (%)" radius={[6, 6, 0, 0]}>
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

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              2. Cumulative Variance Curve (95% Target Line)
            </h3>
            <p className="text-xs text-slate-400">
              Reaches 97.31% at 6 Principal Components (crossing the green 95% line)
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={varianceChartData}
                margin={{ top: 10, right: 20, left: -10, bottom: 5 }}
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
                    value: '95% Threshold',
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

      {/* Row 2: How All 8 Features Contribute + 2D PCA Scatter */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              3. Original Ingredients vs. Selected PCs (k = {selectedK})
            </h3>
            <p className="text-xs text-slate-300 flex items-center gap-1.5 mt-1">
              <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>
                PCA does <strong>not</strong> delete features—it combines all 8 normalized
                ingredients into each component.
              </span>
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={weightedContributions}
                layout="vertical"
                margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" unit="%" tick={{ fill: '#94a3b8', fontSize: 11 }} />
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
                  dataKey="sharePct"
                  name="Share of Retained Variance (%)"
                  fill="#06b6d4"
                  radius={[0, 6, 6, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-bold text-white">
                4. 2D PCA Projection (PC1 vs. PC2)
              </h3>
              <p className="text-xs text-slate-400">
                Data points colored by Concrete Strength (MPa)
              </p>
            </div>
            <div className="flex items-center gap-2.5 text-[11px] font-mono">
              <span className="text-amber-400">● &lt;25 MPa</span>
              <span className="text-cyan-400">● 25–45 MPa</span>
              <span className="text-purple-400">● &gt;45 MPa</span>
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, left: -15, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  type="number"
                  dataKey="pc1"
                  name="PC1"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="pc2"
                  name="PC2"
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
                <Scatter name="Low (<25 MPa)" data={scatterBuckets.low} fill="#f59e0b" opacity={0.7} />
                <Scatter name="Medium (25-45 MPa)" data={scatterBuckets.mid} fill="#06b6d4" opacity={0.7} />
                <Scatter name="High (>45 MPa)" data={scatterBuckets.high} fill="#a855f7" opacity={0.85} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
