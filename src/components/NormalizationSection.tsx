import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  Scale,
  Calculator,
  CheckCircle2,
  BarChart3,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface NormalizationSectionProps {
  pipeline: MLPipelineResult;
  isDetailedView?: boolean;
}

export const NormalizationSection: React.FC<NormalizationSectionProps> = ({
  pipeline,
  isDetailedView = false,
}) => {
  const [activeTab, setActiveTab] = useState<'distributions' | 'dominance' | 'calculator'>(
    'distributions'
  );
  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState<number>(0);
  const [customRawValue, setCustomRawValue] = useState<number>(
    Number(pipeline.scaler.trainMeans[0].toFixed(1))
  );
  const [showTableDrawer, setShowTableDrawer] = useState<boolean>(isDetailedView);

  useEffect(() => {
    setShowTableDrawer(isDetailedView);
  }, [isDetailedView]);

  const meta = pipeline.featureMeta[selectedFeatureIdx];
  const rawStats = pipeline.columnStats[selectedFeatureIdx];
  const trainMean = pipeline.scaler.trainMeans[selectedFeatureIdx];
  const trainStd = pipeline.scaler.trainStds[selectedFeatureIdx];
  const rawDist = rawStats.histogram;
  const normDist = pipeline.scaler.normalizedDistributions[selectedFeatureIdx]?.bins || [];

  const handleSelectFeature = (idx: number) => {
    setSelectedFeatureIdx(idx);
    setCustomRawValue(Number(pipeline.scaler.trainMeans[idx].toFixed(1)));
  };

  const zScore = (customRawValue - trainMean) / trainStd;

  const varianceDominanceData = pipeline.featureMeta.map((f, i) => ({
    name: f.shortName,
    rawVariance: Number(Math.pow(pipeline.scaler.trainStds[i], 2).toFixed(1)),
    scaledVariance: 1.0,
  }));

  return (
    <div className="space-y-6">
      {/* Top 3-Bullet Takeaway Card */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800/90 p-6">
        <h2 className="text-xl font-bold text-white mb-2">
          Why StandardScaler Normalization Is Mandatory Before PCA
        </h2>
        <ul className="space-y-1.5 text-xs sm:text-sm text-slate-300">
          <li className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>
              <strong className="text-white">Vast Scale Differences:</strong> Coarse Aggregate
              averages <strong className="text-amber-300 font-mono">972.9 kg/m³</strong> while
              Superplasticizer averages just{' '}
              <strong className="text-cyan-300 font-mono">6.2 kg/m³</strong>.
            </span>
          </li>
          <li className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0" />
            <span>
              <strong className="text-white">PCA Maximizes Variance:</strong> Without scaling,
              Cement and Aggregates have <strong className="text-rose-300 font-mono">&gt;2,500×</strong>{' '}
              more raw variance than Superplasticizer and would completely hijack PC1.
            </span>
          </li>
          <li className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              <strong className="text-white">Zero Data Leakage:</strong>{' '}
              <code className="text-cyan-300 font-mono">StandardScaler</code> is fit strictly on the{' '}
              <strong className="text-white">824 training rows</strong> so every ingredient has{' '}
              <strong className="text-emerald-300 font-mono">mean = 0</strong> and{' '}
              <strong className="text-emerald-300 font-mono">variance = 1</strong>.
            </span>
          </li>
        </ul>
      </div>

      {/* Clean Sub-Tab Bar */}
      <div className="inline-flex rounded-xl bg-slate-900 border border-slate-800 p-1">
        <button
          onClick={() => setActiveTab('distributions')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'distributions'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <BarChart3 className="w-4 h-4" />
          1. Before vs. After Graphs
        </button>
        <button
          onClick={() => setActiveTab('dominance')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'dominance'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <AlertTriangle className="w-4 h-4" />
          2. Variance Dominance Proof
        </button>
        <button
          onClick={() => setActiveTab('calculator')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeTab === 'calculator'
              ? 'bg-cyan-500 text-slate-950 shadow-sm'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Calculator className="w-4 h-4" />
          3. Live Z-Score Calculator
        </button>
      </div>

      {/* Ingredient Selector Bar (Used across tabs 1 & 3) */}
      {(activeTab === 'distributions' || activeTab === 'calculator' || isDetailedView) && (
        <div className="flex flex-wrap items-center gap-2 bg-slate-900/70 border border-slate-800/80 p-3 rounded-xl">
          <span className="text-xs font-semibold text-slate-400 px-2">Select Ingredient:</span>
          {pipeline.featureMeta.map((f, idx) => (
            <button
              key={f.key}
              onClick={() => handleSelectFeature(idx)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                selectedFeatureIdx === idx
                  ? 'bg-cyan-500 text-slate-950 font-bold'
                  : 'bg-slate-950 text-slate-300 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              {f.shortName}
            </button>
          ))}
        </div>
      )}

      {/* TAB 1: Side-by-Side Before & After Normalization Graphs */}
      {(activeTab === 'distributions' || isDetailedView) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Before Normalization */}
          <div className="rounded-2xl bg-slate-900/80 border border-amber-500/30 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-amber-400">
                  Before Normalization (Raw Units)
                </span>
                <h3 className="text-base font-bold text-white mt-0.5">
                  {meta.name} ({meta.unit})
                </h3>
              </div>
              <div className="text-right font-mono text-xs">
                <div className="text-amber-300">μ = {trainMean.toFixed(2)}</div>
                <div className="text-slate-400">σ = {trainStd.toFixed(2)}</div>
              </div>
            </div>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rawDist} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="label"
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                    angle={-20}
                    textAnchor="end"
                    height={45}
                  />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="count" name="Raw Count" fill="#f59e0b" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* After Normalization */}
          <div className="rounded-2xl bg-slate-900/80 border border-cyan-500/30 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-cyan-400">
                  After StandardScaler (Z-Score)
                </span>
                <h3 className="text-base font-bold text-white mt-0.5">
                  {meta.name} (Standardized)
                </h3>
              </div>
              <div className="text-right font-mono text-xs">
                <div className="text-cyan-300">μ = 0.00</div>
                <div className="text-emerald-400">σ = 1.00</div>
              </div>
            </div>

            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={normDist} margin={{ top: 10, right: 10, left: -10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="label"
                    tick={{ fill: '#94a3b8', fontSize: 10 }}
                    angle={-20}
                    textAnchor="end"
                    height={45}
                  />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="count" name="Scaled Count" fill="#06b6d4" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Variance Dominance Comparison */}
      {(activeTab === 'dominance' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              Raw Variance (σ²) Before Scaling vs. Equalized Variance After Scaling
            </h3>
            <p className="text-xs text-slate-400">
              Before normalization, Cement (σ² = 11,646) and Fine Aggregate (σ² = 6,419) dwarf
              Superplasticizer (σ² = 35.8). After StandardScaler, every ingredient has variance =
              1.0.
            </p>
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={varianceDominanceData}
                margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    fontSize: '12px',
                  }}
                />
                <Bar
                  dataKey="rawVariance"
                  name="Unscaled Variance (σ²)"
                  fill="#f43f5e"
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* TAB 3: Interactive Live Z-Score Calculator */}
      {(activeTab === 'calculator' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-5">
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white">
              Interactive StandardScaler Formula Simulator ({meta.name})
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">
                  Adjust Raw {meta.shortName} ({meta.unit}):
                </span>
                <span className="font-mono font-bold text-amber-300 text-base">
                  {customRawValue.toFixed(1)} {meta.unit}
                </span>
              </div>
              <input
                type="range"
                min={rawStats.min}
                max={rawStats.max}
                step={0.5}
                value={customRawValue}
                onChange={(e) => setCustomRawValue(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="flex justify-between text-[11px] font-mono text-slate-500">
                <span>Min: {rawStats.min.toFixed(1)}</span>
                <span>Train Mean: {trainMean.toFixed(1)}</span>
                <span>Max: {rawStats.max.toFixed(1)}</span>
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-center space-y-2">
              <div className="text-xs text-slate-400">z = (x - μ_train) / σ_train</div>
              <div className="text-sm text-slate-200">
                z = ({customRawValue.toFixed(2)} - {trainMean.toFixed(2)}) / {trainStd.toFixed(2)}
              </div>
              <div className="text-2xl font-bold text-cyan-400 pt-1">
                z = {zScore >= 0 ? '+' : ''}
                {zScore.toFixed(4)} σ
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Expandable Drawer for All 8 Feature Scaling Parameters */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowTableDrawer(!showTableDrawer)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
        >
          <span className="text-sm font-semibold text-slate-200">
            All 8 Ingredients — StandardScaler Parameters (μ_train, σ_train) &amp; Sample Values
          </span>
          <span className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            {showTableDrawer ? 'Hide Table' : 'Show Full Table'}
            {showTableDrawer ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </span>
        </button>

        {showTableDrawer && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 text-left font-sans">Ingredient</th>
                  <th className="py-2 text-right">Train Mean (μ)</th>
                  <th className="py-2 text-right">Train Std (σ)</th>
                  <th className="py-2 text-right">Sample #1 Raw</th>
                  <th className="py-2 text-right text-cyan-300">Sample #1 Scaled (z)</th>
                </tr>
              </thead>
              <tbody>
                {pipeline.featureMeta.map((f, idx) => {
                  const rawVal = pipeline.scaler.sampleTransformations[0]?.raw[idx] ?? 0;
                  const scaledVal = pipeline.scaler.sampleTransformations[0]?.scaled[idx] ?? 0;
                  return (
                    <tr key={f.key} className="border-b border-slate-800/50">
                      <td className="py-2 font-sans font-medium text-slate-200">
                        {f.name} ({f.unit})
                      </td>
                      <td className="py-2 text-right text-amber-300">
                        {pipeline.scaler.trainMeans[idx].toFixed(3)}
                      </td>
                      <td className="py-2 text-right text-slate-300">
                        {pipeline.scaler.trainStds[idx].toFixed(3)}
                      </td>
                      <td className="py-2 text-right text-slate-400">{rawVal.toFixed(2)}</td>
                      <td className="py-2 text-right text-cyan-300 font-semibold">
                        {scaledVal >= 0 ? '+' : ''}
                        {scaledVal.toFixed(4)}
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
