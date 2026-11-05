import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Database, Scale, CheckCircle2, Grid, BarChart3, Table2 } from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface DatasetExplorerSectionProps {
  pipeline: MLPipelineResult;
}

export const DatasetExplorerSection: React.FC<DatasetExplorerSectionProps> = ({ pipeline }) => {
  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState<number>(0);
  const [tableMode, setTableMode] = useState<'describe' | 'head'>('describe');

  const meta = pipeline.featureMeta[selectedFeatureIdx];
  const rawStats = pipeline.columnStats[selectedFeatureIdx];
  const trainMean = pipeline.scaler.trainMeans[selectedFeatureIdx];
  const trainStd = pipeline.scaler.trainStds[selectedFeatureIdx];
  const rawDist = rawStats.histogram;
  const normDist = pipeline.scaler.normalizedDistributions[selectedFeatureIdx]?.bins || [];

  // Target column (Concrete compressive strength) histogram + Gaussian KDE approximation (Cell 5)
  const strengthHistWithKDE = useMemo(() => {
    const targetStat = pipeline.columnStats[8];
    const vals = pipeline.rawRows.map((r) => r[8]);
    const n = vals.length;
    const std = targetStat.std || 1;
    // Silverman's rule of thumb bandwidth
    const bw = 1.06 * std * Math.pow(n, -0.2);
    const binWidth =
      targetStat.histogram.length > 0
        ? targetStat.histogram[0].binEnd - targetStat.histogram[0].binStart
        : 4.5;

    return targetStat.histogram.map((bin) => {
      const center = (bin.binStart + bin.binEnd) / 2;
      let densitySum = 0;
      for (let i = 0; i < n; i++) {
        const u = (center - vals[i]) / bw;
        densitySum += Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
      }
      const density = densitySum / (n * bw);
      const kdeFrequency = density * n * binWidth;
      return {
        label: `${bin.binStart.toFixed(0)}–${bin.binEnd.toFixed(0)}`,
        binCenter: Number(center.toFixed(1)),
        count: bin.count,
        kde: Number(kdeFrequency.toFixed(1)),
      };
    });
  }, [pipeline]);

  // Short labels for the 9x9 correlation matrix (Cell 4)
  const colShortLabels = useMemo(
    () => [...pipeline.featureMeta.map((f) => f.shortName), 'Strength'],
    [pipeline.featureMeta]
  );

  const getCoolwarmStyle = (val: number) => {
    // Coolwarm palette approximation: negative -> blue/cyan, near 0 -> slate, positive -> red/rose
    const clamped = Math.max(-1, Math.min(1, val));
    if (clamped >= 0.99) {
      return 'bg-rose-500/40 text-white font-bold border border-rose-400/40';
    }
    if (clamped >= 0.35) {
      return 'bg-rose-500/30 text-rose-200 font-semibold';
    }
    if (clamped >= 0.15) {
      return 'bg-rose-500/15 text-rose-200';
    }
    if (clamped <= -0.35) {
      return 'bg-blue-500/30 text-blue-200 font-semibold';
    }
    if (clamped <= -0.15) {
      return 'bg-blue-500/15 text-blue-200';
    }
    return 'bg-slate-900/90 text-slate-400';
  };

  return (
    <div className="space-y-6">
      {/* 4 Top Summary Cards matching Cells 2, 3, 6 */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Dataset Shape</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">
            ({pipeline.totalRows}, 9)
          </div>
          <div className="text-xs text-slate-500 mt-1">
            8 float64, 1 int64 (Age)
          </div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Train / Test Split (80/20)</div>
          <div className="text-2xl font-bold font-mono text-cyan-400 mt-1">
            ({pipeline.trainCount}, 8) / ({pipeline.testCount}, 8)
          </div>
          <div className="text-xs text-slate-500 mt-1">random_state = 42</div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Target Variable (y)</div>
          <div className="text-2xl font-bold font-mono text-purple-400 mt-1">Strength (MPa)</div>
          <div className="text-xs text-slate-500 mt-1">Range: 2.33 to 82.60 MPa</div>
        </div>

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-5">
          <div className="text-xs text-slate-400">Missing Values (df.isnull().sum())</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">0 across all 9 cols</div>
          <div className="text-xs text-slate-500 mt-1">100% complete dataset</div>
        </div>
      </div>

      {/* Part 1: Statistical Summary (df.describe()) & First 5 Rows (df.head()) - Cells 2 & 3 */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <Database className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-lg font-bold text-white">
                1. Dataset Exploration &amp; Statistical Summary
              </h2>
              <p className="text-xs text-slate-400">
                Inspect <span className="font-mono text-cyan-300">df.describe()</span> summary statistics or{' '}
                <span className="font-mono text-cyan-300">df.head()</span> first 5 concrete mix samples
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setTableMode('describe')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                tableMode === 'describe'
                  ? 'bg-cyan-500 text-slate-950'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              <Table2 className="w-3.5 h-3.5" />
              Statistical Summary (df.describe)
            </button>
            <button
              onClick={() => setTableMode('head')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                tableMode === 'head'
                  ? 'bg-cyan-500 text-slate-950'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              First 5 Rows (df.head)
            </button>
          </div>
        </div>

        {tableMode === 'describe' ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-3 text-left font-sans">Variable</th>
                  <th className="py-2.5 px-2 text-right">count</th>
                  <th className="py-2.5 px-2 text-right">mean</th>
                  <th className="py-2.5 px-2 text-right">std</th>
                  <th className="py-2.5 px-2 text-right">min</th>
                  <th className="py-2.5 px-2 text-right">25%</th>
                  <th className="py-2.5 px-2 text-right">50%</th>
                  <th className="py-2.5 px-2 text-right">75%</th>
                  <th className="py-2.5 px-2 text-right">max</th>
                </tr>
              </thead>
              <tbody>
                {pipeline.columnStats.map((stat, idx) => {
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
                        <span className="text-slate-500 ml-1.5 text-[11px]">({stat.unit})</span>
                      </td>
                      <td className="py-2.5 px-2 text-right text-slate-400">{stat.count}</td>
                      <td className="py-2.5 px-2 text-right text-white">{stat.mean.toFixed(2)}</td>
                      <td className="py-2.5 px-2 text-right text-cyan-300">{stat.std.toFixed(2)}</td>
                      <td className="py-2.5 px-2 text-right text-slate-300">{stat.min.toFixed(2)}</td>
                      <td className="py-2.5 px-2 text-right text-slate-300">{stat.q25.toFixed(2)}</td>
                      <td className="py-2.5 px-2 text-right text-emerald-300">{stat.median.toFixed(2)}</td>
                      <td className="py-2.5 px-2 text-right text-slate-300">{stat.q75.toFixed(2)}</td>
                      <td className="py-2.5 px-2 text-right text-slate-200">{stat.max.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-2.5 text-left font-sans">Row</th>
                  {pipeline.columnStats.map((c, idx) => (
                    <th
                      key={c.key}
                      className={`py-2.5 px-2.5 text-right font-sans ${
                        idx === 8 ? 'text-purple-300' : ''
                      }`}
                    >
                      {c.shortName}
                      <div className="text-[10px] font-normal text-slate-500">({c.unit})</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pipeline.rawRows.slice(0, 5).map((row, rIdx) => (
                  <tr key={rIdx} className="border-b border-slate-800/60 hover:bg-slate-800/40">
                    <td className="py-2.5 px-2.5 text-slate-400 font-bold">#{rIdx}</td>
                    {row.map((val, cIdx) => (
                      <td
                        key={cIdx}
                        className={`py-2.5 px-2.5 text-right ${
                          cIdx === 8
                            ? 'text-purple-300 font-bold bg-purple-500/10'
                            : 'text-slate-200'
                        }`}
                      >
                        {cIdx === 7 ? val.toFixed(0) : val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Part 2: 9x9 Annotated Correlation Heatmap & Concrete Strength Distribution (Cells 4 & 5) */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Cell 4: Correlation Heatmap */}
        <div className="xl:col-span-7 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <Grid className="w-5 h-5 text-rose-400" />
              <div>
                <h2 className="text-base font-bold text-white">
                  2. Feature Correlation Matrix (9×9 Annotated Heatmap)
                </h2>
                <p className="text-xs text-slate-400">
                  Pearson correlation coefficients (<span className="font-mono">df.corr()</span>) across all 8 inputs and Strength
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-[11px] font-mono">
              <span className="px-2 py-0.5 rounded bg-blue-500/25 text-blue-200">- Negative</span>
              <span className="px-2 py-0.5 rounded bg-rose-500/25 text-rose-200">+ Positive</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-[11px] font-mono border-collapse">
              <thead>
                <tr>
                  <th className="p-1.5 text-left text-slate-400 font-sans">Variable</th>
                  {colShortLabels.map((lbl, i) => (
                    <th
                      key={i}
                      className="p-1.5 text-center text-slate-300 font-sans whitespace-nowrap"
                    >
                      {lbl}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pipeline.correlationMatrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="p-1.5 font-sans font-semibold text-slate-300 whitespace-nowrap">
                      {colShortLabels[rIdx]}
                    </td>
                    {row.map((val, cIdx) => (
                      <td key={cIdx} className="p-1">
                        <div
                          className={`w-full py-1.5 px-1 rounded text-center transition-all ${getCoolwarmStyle(
                            val
                          )}`}
                        >
                          {val.toFixed(2)}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Cell 5: Concrete Strength Distribution (Histogram + KDE) */}
        <div className="xl:col-span-5 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 flex flex-col justify-between space-y-4">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <div>
              <h2 className="text-base font-bold text-white">
                3. Concrete Strength Distribution (Histogram + KDE)
              </h2>
              <p className="text-xs text-slate-400">
                Target distribution (<span className="font-mono">sns.histplot(kde=True)</span>) across 1,030 mixes
              </p>
            </div>
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={strengthHistWithKDE}
                margin={{ top: 10, right: 15, left: -15, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  angle={-25}
                  textAnchor="end"
                  height={48}
                />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  label={{
                    value: 'Frequency',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#64748b',
                    fontSize: 11,
                  }}
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
                  dataKey="count"
                  name="Frequency (Samples)"
                  fill="#8b5cf6"
                  fillOpacity={0.65}
                  radius={[4, 4, 0, 0]}
                />
                <Line
                  type="monotone"
                  dataKey="kde"
                  name="KDE Curve"
                  stroke="#38bdf8"
                  strokeWidth={2.5}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-2 pt-1 text-center font-mono text-xs">
            <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">Mean Strength</div>
              <div className="text-sm font-bold text-purple-300">
                {pipeline.columnStats[8].mean.toFixed(2)} MPa
              </div>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">Median (50%)</div>
              <div className="text-sm font-bold text-cyan-300">
                {pipeline.columnStats[8].median.toFixed(2)} MPa
              </div>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
              <div className="text-[10px] text-slate-400">Std Dev</div>
              <div className="text-sm font-bold text-slate-200">
                {pipeline.columnStats[8].std.toFixed(2)} MPa
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Part 3: StandardScaler Normalization (Cell 7) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Scale className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-bold text-white">
                4. Feature Standardization (StandardScaler before PCA)
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-300 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                <strong className="text-white">StandardScaler</strong> transforms each training feature
                to <strong className="text-cyan-300 font-mono">Mean = 0, Std = 1</strong> so features
                with large numerical ranges (like Coarse Aggregate) do not dominate PCA.
              </span>
            </p>
          </div>
        </div>

        {/* Ingredient Selector */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-xs font-semibold text-slate-400 mr-1">Select Feature:</span>
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
                  Before Scaling (Raw Feature)
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

            <div className="h-52">
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
                  After StandardScaler (Z-Score)
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

            <div className="h-52">
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
