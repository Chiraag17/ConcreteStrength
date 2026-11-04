import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Database, Scale, CheckCircle2 } from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface DatasetExplorerSectionProps {
  pipeline: MLPipelineResult;
}

export const DatasetExplorerSection: React.FC<DatasetExplorerSectionProps> = ({ pipeline }) => {
  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState<number>(0);

  const meta = pipeline.featureMeta[selectedFeatureIdx];
  const rawStats = pipeline.columnStats[selectedFeatureIdx];
  const trainMean = pipeline.scaler.trainMeans[selectedFeatureIdx];
  const trainStd = pipeline.scaler.trainStds[selectedFeatureIdx];
  const rawDist = rawStats.histogram;
  const normDist = pipeline.scaler.normalizedDistributions[selectedFeatureIdx]?.bins || [];

  // Correlation of each column (0..7) with Target Strength (column 8)
  const strengthCorrelations = pipeline.correlationMatrix[8] || [];

  return (
    <div className="space-y-6">
      {/* 4 Simple Top Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Total Dataset Records</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">
            {pipeline.totalRows.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {pipeline.trainCount} Train / {pipeline.testCount} Test (80/20)
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Input Features</div>
          <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">8 Ingredients</div>
          <div className="text-xs text-slate-500 mt-1">Measured in kg/m³ &amp; days</div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Target Variable</div>
          <div className="text-2xl font-bold font-mono text-purple-400 mt-1">Strength (MPa)</div>
          <div className="text-xs text-slate-500 mt-1">Range: 2.33 to 82.60 MPa</div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Missing Values</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">0</div>
          <div className="text-xs text-slate-500 mt-1">100% complete dataset</div>
        </div>
      </div>

      {/* Part 1: Clean Summary Table of the 8 Concrete Ingredients + Target */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <Database className="w-5 h-5 text-cyan-400" />
          <div>
            <h2 className="text-lg font-bold text-white">
              1. Concrete Ingredients &amp; Dataset Summary
            </h2>
            <p className="text-xs text-slate-400">
              Overview of all 8 input features and their correlation with Concrete Compressive Strength
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 text-left font-sans">Variable</th>
                <th className="py-2.5 px-3 text-left font-sans">Unit</th>
                <th className="py-2.5 px-3 text-right">Min</th>
                <th className="py-2.5 px-3 text-right">Mean</th>
                <th className="py-2.5 px-3 text-right">Max</th>
                <th className="py-2.5 px-3 text-right">Std Dev</th>
                <th className="py-2.5 px-3 text-right">Correlation with Strength</th>
              </tr>
            </thead>
            <tbody>
              {pipeline.columnStats.map((stat, idx) => {
                const corr = strengthCorrelations[idx] ?? 1;
                const isTarget = idx === 8;
                return (
                  <tr
                    key={stat.key}
                    className={`border-b border-slate-800/60 ${
                      isTarget ? 'bg-purple-500/10 font-semibold' : 'hover:bg-slate-800/40'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans text-slate-100">
                      {idx < 8 ? `${idx + 1}. ${stat.name}` : `Target: ${stat.name}`}
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-400">{stat.unit}</td>
                    <td className="py-2.5 px-3 text-right text-slate-300">{stat.min.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right text-white">{stat.mean.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-300">{stat.max.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right text-cyan-300">{stat.std.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-right">
                      {isTarget ? (
                        <span className="text-purple-300">1.00 (Target)</span>
                      ) : (
                        <span
                          className={`px-2 py-0.5 rounded ${
                            corr >= 0.25
                              ? 'bg-emerald-500/15 text-emerald-300'
                              : corr <= -0.15
                              ? 'bg-rose-500/15 text-rose-300'
                              : 'text-slate-400'
                          }`}
                        >
                          {corr >= 0 ? '+' : ''}
                          {corr.toFixed(2)}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Part 2: Interactive Normalization Simulator (Before vs After StandardScaler) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Scale className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">
                2. Why We Normalize Before PCA (StandardScaler)
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-300 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                Ingredients have very different scales (Coarse Aggregate is ~973 kg/m³ while
                Superplasticizer is ~6.2 kg/m³).{' '}
                <strong className="text-white">StandardScaler</strong> centers every ingredient to{' '}
                <strong className="text-cyan-300 font-mono">Mean = 0, Std = 1</strong> so each
                ingredient contributes fairly to PCA.
              </span>
            </p>
          </div>
        </div>

        {/* Ingredient Selector */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-xs font-semibold text-slate-400 mr-1">Select Ingredient:</span>
          {pipeline.featureMeta.map((f, idx) => (
            <button
              key={f.key}
              onClick={() => setSelectedFeatureIdx(idx)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                selectedFeatureIdx === idx
                  ? 'bg-cyan-500 text-slate-950 shadow-sm'
                  : 'bg-slate-950 text-slate-300 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              {f.shortName}
            </button>
          ))}
        </div>

        {/* Side-by-Side Before & After Graphs */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-xl bg-slate-950/80 border border-amber-500/30 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[11px] font-mono uppercase text-amber-400">
                  Before Normalization (Raw Values)
                </div>
                <div className="text-sm font-bold text-white mt-0.5">
                  {meta.name} ({meta.unit})
                </div>
              </div>
              <div className="text-right font-mono text-xs">
                <div className="text-amber-300">Mean = {trainMean.toFixed(1)}</div>
                <div className="text-slate-400">Std = {trainStd.toFixed(1)}</div>
              </div>
            </div>

            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rawDist} margin={{ top: 10, right: 10, left: -15, bottom: 20 }}>
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
                  <Bar dataKey="count" name="Samples" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/80 border border-cyan-500/30 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[11px] font-mono uppercase text-cyan-400">
                  After Normalization (StandardScaler Z-Score)
                </div>
                <div className="text-sm font-bold text-white mt-0.5">
                  {meta.name} (Scaled)
                </div>
              </div>
              <div className="text-right font-mono text-xs">
                <div className="text-cyan-300">Mean = 0.0</div>
                <div className="text-emerald-400">Std = 1.0</div>
              </div>
            </div>

            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={normDist} margin={{ top: 10, right: 10, left: -15, bottom: 20 }}>
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
                  <Bar dataKey="count" name="Samples" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
