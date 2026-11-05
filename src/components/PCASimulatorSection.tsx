import React, { useMemo } from 'react';
import {
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  ComposedChart,
} from 'recharts';
import { Sliders, Table2, Sigma, Layers } from 'lucide-react';
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
  const dimsReduced = 8 - selectedK;
  const dimRedPct = currentKMetrics.dimensionalityReductionPercent;

  // Cell 8 & 9: variance_table data
  const varianceTableData = useMemo(() => {
    return pipeline.pca.explainedVarianceRatio.map((ratio, idx) => {
      const pcNum = idx + 1;
      return {
        pcNum,
        pc: `PC${pcNum}`,
        individualPct: Number((ratio * 100).toFixed(2)),
        cumulativePct: Number((pipeline.pca.cumulativeVarianceRatio[idx] * 100).toFixed(2)),
        isRetained: pcNum <= selectedK,
      };
    });
  }, [pipeline.pca, selectedK]);

  // Cell 13 & 14: PCA Component Coefficients & Mathematical Equations for PC1..PC(selectedK)
  const selectedComponents = useMemo(
    () => pipeline.pca.components.slice(0, selectedK),
    [pipeline.pca.components, selectedK]
  );

  // Cell 15: First 5 rows of PCA Transformed Dataset (pca_df.head())
  const pcaHeadRows = useMemo(
    () => (pipeline.pca.trainTransformedHead || []).slice(0, 5),
    [pipeline.pca.trainTransformedHead]
  );

  return (
    <div className="space-y-6">
      {/* Main Interactive PCA Control & Summary Card (Cells 10, 11, 12) */}
      <div className="rounded-2xl bg-slate-900/90 border border-cyan-500/35 p-6 space-y-5 shadow-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <Sliders className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-xl font-bold text-white">
                Selected Principal Components:{' '}
                <span className="text-cyan-400 font-mono">{selectedK}</span> of 8
              </h2>
              <p className="text-xs text-slate-400">
                Notebook selects <span className="font-mono text-emerald-300">n = 6</span> components to retain ≥95% variance (97.31%)
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
            <span>1 PC (28.02%)</span>
            <span className="text-emerald-400 font-semibold">
              ★ 6 PCs = 97.31% Variance Retained (Notebook Default)
            </span>
            <span>8 PCs (100.00%)</span>
          </div>
        </div>

        {/* 4 Summary Cards matching Cells 10 & 12 outputs */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 pt-1">
          <div className="rounded-xl bg-slate-950/90 border border-slate-800 p-4">
            <div className="text-xs text-slate-400">Original → Selected Features</div>
            <div className="text-2xl font-bold font-mono text-white mt-1">
              8 → <span className="text-cyan-400">{selectedK}</span>
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Dimensions Reduced: {dimsReduced} ({dimRedPct.toFixed(1)}%)
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/90 border border-cyan-500/30 p-4">
            <div className="text-xs text-slate-400">Variance Retained</div>
            <div className="text-2xl font-bold font-mono text-cyan-300 mt-1">
              {retainedPct.toFixed(2)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {retainedPct >= 95 ? 'Meets ≥95% target' : 'Below 95% target'}
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/90 border border-slate-800 p-4">
            <div className="text-xs text-slate-400">Before PCA Shape</div>
            <div className="text-2xl font-bold font-mono text-slate-200 mt-1">
              ({pipeline.trainCount}, 8)
            </div>
            <div className="text-xs text-slate-500 mt-1">X_train_scaled.shape</div>
          </div>

          <div className="rounded-xl bg-slate-950/90 border border-purple-500/30 p-4">
            <div className="text-xs text-slate-400">After PCA Shape</div>
            <div className="text-2xl font-bold font-mono text-purple-300 mt-1">
              ({pipeline.trainCount}, {selectedK})
            </div>
            <div className="text-xs text-slate-500 mt-1">X_train_pca.shape</div>
          </div>
        </div>
      </div>

      {/* Row 1: Variance Table (Cell 8) & Cumulative Explained Variance Chart (Cell 9) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Cell 8: variance_table */}
        <div className="lg:col-span-5 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div className="flex items-center gap-2.5">
            <Table2 className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-base font-bold text-white">
                1. PCA Explained Variance Table
              </h3>
              <p className="text-xs text-slate-400">
                Exact <span className="font-mono text-cyan-300">variance_table</span> output for PC1–PC8
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 px-3 text-left font-sans">Principal Component</th>
                  <th className="py-2 px-3 text-right">Explained (%)</th>
                  <th className="py-2 px-3 text-right">Cumulative (%)</th>
                </tr>
              </thead>
              <tbody>
                {varianceTableData.map((row) => (
                  <tr
                    key={row.pc}
                    className={`border-b border-slate-800/60 ${
                      row.pcNum === selectedK
                        ? 'bg-cyan-500/15 font-bold'
                        : row.isRetained
                        ? 'hover:bg-slate-800/40'
                        : 'opacity-50'
                    }`}
                  >
                    <td className="py-2 px-3 font-sans flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          row.isRetained ? 'bg-cyan-400' : 'bg-slate-600'
                        }`}
                      />
                      <span className="text-slate-100">{row.pc}</span>
                      {row.pcNum === 6 && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono">
                          ≥95%
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right text-cyan-300">
                      {row.individualPct.toFixed(2)}%
                    </td>
                    <td className="py-2 px-3 text-right text-purple-300">
                      {row.cumulativePct.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Cell 9: PCA - Cumulative Explained Variance Plot */}
        <div className="lg:col-span-7 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              2. PCA — Cumulative Explained Variance Curve
            </h3>
            <p className="text-xs text-slate-400">
              Cumulative variance across 1–8 components with the dashed <span className="text-rose-400 font-semibold">95% Variance</span> threshold line
            </p>
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={varianceTableData}
                margin={{ top: 15, right: 20, left: -10, bottom: 5 }}
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
                  stroke="#f43f5e"
                  strokeDasharray="5 5"
                  label={{
                    value: '95% Variance',
                    fill: '#f43f5e',
                    fontSize: 11,
                    position: 'insideBottomRight',
                  }}
                />
                <Bar
                  dataKey="individualPct"
                  name="Individual Explained Variance (%)"
                  radius={[5, 5, 0, 0]}
                >
                  {varianceTableData.map((entry, idx) => (
                    <Cell
                      key={`cell-${idx}`}
                      fill={entry.isRetained ? '#06b6d4' : '#334155'}
                      fillOpacity={0.55}
                    />
                  ))}
                </Bar>
                <Line
                  type="monotone"
                  dataKey="cumulativePct"
                  name="Cumulative Variance (%)"
                  stroke="#38bdf8"
                  strokeWidth={3}
                  dot={{ r: 5, fill: '#38bdf8' }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Row 2: PCA Component Coefficients Table (Cell 13) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <Layers className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-base font-bold text-white">
                3. PCA Component Coefficients (<span className="font-mono text-cyan-300">pca_selected.components_</span>)
              </h3>
              <p className="text-xs text-slate-400">
                Feature loadings showing how each of the 8 standardized ingredients is weighted inside PC1 through PC{selectedK}
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Rounded to 3 decimal places
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 text-left font-sans">Component</th>
                {pipeline.featureMeta.map((f) => (
                  <th key={f.key} className="py-2.5 px-2.5 text-right font-sans">
                    {f.shortName}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {selectedComponents.map((compVec, pcIdx) => (
                <tr
                  key={pcIdx}
                  className="border-b border-slate-800/60 hover:bg-slate-800/40"
                >
                  <td className="py-2.5 px-3 font-bold text-cyan-300">PC{pcIdx + 1}</td>
                  {compVec.map((weight, featIdx) => {
                    const rounded = Number(weight.toFixed(3));
                    const isStrongPos = rounded >= 0.3;
                    const isStrongNeg = rounded <= -0.3;
                    return (
                      <td key={featIdx} className="py-2.5 px-2.5 text-right">
                        <span
                          className={`px-2 py-0.5 rounded ${
                            isStrongPos
                              ? 'bg-emerald-500/20 text-emerald-300 font-semibold'
                              : isStrongNeg
                              ? 'bg-rose-500/20 text-rose-300 font-semibold'
                              : 'text-slate-300'
                          }`}
                        >
                          {rounded >= 0 ? '+' : ''}
                          {rounded.toFixed(3)}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row 3: Explicit PCA Mathematical Equations (Cell 14) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <Sigma className="w-5 h-5 text-purple-400" />
          <div>
            <h3 className="text-base font-bold text-white">
              4. PCA Mathematical Equations (PC1 – PC{selectedK})
            </h3>
            <p className="text-xs text-slate-400">
              Linear combination equations printed in the Colab notebook for each Principal Component
            </p>
          </div>
        </div>

        <div className="space-y-2.5">
          {selectedComponents.map((compVec, pcIdx) => (
            <div
              key={pcIdx}
              className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800/90 font-mono text-xs leading-relaxed overflow-x-auto"
            >
              <span className="text-cyan-400 font-bold mr-2">PC{pcIdx + 1} =</span>
              {compVec.map((weight, featIdx) => {
                const featName = pipeline.featureMeta[featIdx].shortName;
                const wStr = weight.toFixed(3);
                return (
                  <React.Fragment key={featIdx}>
                    <span className="text-slate-300">
                      (<span className={weight >= 0 ? 'text-emerald-300' : 'text-rose-300'}>{wStr}</span>
                      {' × '}
                      <span className="text-white">{featName}</span>)
                    </span>
                    {featIdx < compVec.length - 1 && (
                      <span className="text-slate-500 mx-1.5">+</span>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Row 4: PCA Transformed Dataset Preview (pca_df.head(), Cell 15) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <Table2 className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-base font-bold text-white">
                5. PCA Transformed Dataset Preview (<span className="font-mono text-emerald-300">pca_df.head()</span>)
              </h3>
              <p className="text-xs text-slate-400">
                First 5 standardized training samples projected onto PC1 through PC{selectedK}
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Shape: ({pipeline.trainCount}, {selectedK})
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3 text-left font-sans">Sample Index</th>
                {Array.from({ length: selectedK }, (_, i) => (
                  <th key={i} className="py-2.5 px-3 text-right text-cyan-300">
                    PC{i + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pcaHeadRows.map((pcRow, rIdx) => (
                <tr
                  key={rIdx}
                  className="border-b border-slate-800/60 hover:bg-slate-800/40"
                >
                  <td className="py-2.5 px-3 text-slate-400 font-bold">Row {rIdx}</td>
                  {pcRow.slice(0, selectedK).map((val, cIdx) => (
                    <td key={cIdx} className="py-2.5 px-3 text-right text-slate-200">
                      {val.toFixed(6)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
