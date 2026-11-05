import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
  Legend,
} from 'recharts';
import { GitCompare, Table2, BarChart3, Award } from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface RegressionComparisonSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
}

export const RegressionComparisonSection: React.FC<RegressionComparisonSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
}) => {
  const modelA = pipeline.regression.withoutPCA;
  const modelB = pipeline.regression.withPCAByK[selectedK];

  // Cell 18: Model Performance Matrix rows
  const performanceMatrixRows = useMemo(
    () => [
      {
        metric: 'Input features',
        withoutPCA: modelA.featuresCount.toFixed(0),
        withPCA: modelB.featuresCount.toFixed(0),
        diff: `${modelB.featuresCount - modelA.featuresCount} features`,
      },
      {
        metric: 'Variance retained (%)',
        withoutPCA: (modelA.retainedVarianceRatio * 100).toFixed(4),
        withPCA: (modelB.retainedVarianceRatio * 100).toFixed(4),
        diff: `${((modelB.retainedVarianceRatio - 1) * 100).toFixed(2)}%`,
      },
      {
        metric: 'MAE (Mean Absolute Error)',
        withoutPCA: modelA.mae.toFixed(4),
        withPCA: modelB.mae.toFixed(4),
        diff: `${modelB.mae >= modelA.mae ? '+' : ''}${(modelB.mae - modelA.mae).toFixed(4)}`,
      },
      {
        metric: 'MSE (Mean Squared Error)',
        withoutPCA: modelA.mse.toFixed(4),
        withPCA: modelB.mse.toFixed(4),
        diff: `${modelB.mse >= modelA.mse ? '+' : ''}${(modelB.mse - modelA.mse).toFixed(4)}`,
      },
      {
        metric: 'RMSE (Root Mean Squared Error)',
        withoutPCA: modelA.rmse.toFixed(4),
        withPCA: modelB.rmse.toFixed(4),
        diff: `${modelB.rmse >= modelA.rmse ? '+' : ''}${(modelB.rmse - modelA.rmse).toFixed(4)}`,
      },
      {
        metric: 'R2 Score',
        withoutPCA: modelA.r2.toFixed(4),
        withPCA: modelB.r2.toFixed(4),
        diff: `${modelB.r2 >= modelA.r2 ? '+' : ''}${(modelB.r2 - modelA.r2).toFixed(4)}`,
      },
    ],
    [modelA, modelB]
  );

  // Cell 19: 2x2 Subplots data for MAE, MSE, RMSE, R2 Score
  const metricSubplots = useMemo(
    () => [
      {
        key: 'mae',
        title: 'MAE (Mean Absolute Error)',
        subtitle: 'Lower is better (MPa)',
        data: [
          { model: 'Without PCA', value: Number(modelA.mae.toFixed(3)), color: '#10b981' },
          { model: `With PCA (k=${selectedK})`, value: Number(modelB.mae.toFixed(3)), color: '#06b6d4' },
        ],
      },
      {
        key: 'mse',
        title: 'MSE (Mean Squared Error)',
        subtitle: 'Lower is better (MPa²)',
        data: [
          { model: 'Without PCA', value: Number(modelA.mse.toFixed(3)), color: '#10b981' },
          { model: `With PCA (k=${selectedK})`, value: Number(modelB.mse.toFixed(3)), color: '#06b6d4' },
        ],
      },
      {
        key: 'rmse',
        title: 'RMSE (Root Mean Squared Error)',
        subtitle: 'Lower is better (MPa)',
        data: [
          { model: 'Without PCA', value: Number(modelA.rmse.toFixed(3)), color: '#10b981' },
          { model: `With PCA (k=${selectedK})`, value: Number(modelB.rmse.toFixed(3)), color: '#06b6d4' },
        ],
      },
      {
        key: 'r2',
        title: 'R2 Score (Coefficient of Determination)',
        subtitle: 'Higher is better (max 1.0)',
        data: [
          { model: 'Without PCA', value: Number(modelA.r2.toFixed(4)), color: '#10b981' },
          { model: `With PCA (k=${selectedK})`, value: Number(modelB.r2.toFixed(4)), color: '#06b6d4' },
        ],
      },
    ],
    [modelA, modelB, selectedK]
  );

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

  return (
    <div className="space-y-6">
      {/* Part 1: Model Performance Matrix Table (Cells 16, 17, 18) */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800 p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <Table2 className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-xl font-bold text-white">
                1. Model Performance Matrix (Without PCA vs. With PCA)
              </h2>
              <p className="text-xs text-slate-400">
                Exact <span className="font-mono text-cyan-300">results</span> DataFrame comparing Linear Regression on 206 test samples
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs text-slate-400 mr-1">Select PCs (k):</span>
            {[1, 2, 3, 4, 5, 6, 7, 8].map((k) => (
              <button
                key={k}
                onClick={() => onSelectK(k)}
                className={`w-8 h-8 rounded-lg text-xs font-mono font-bold transition-all ${
                  selectedK === k
                    ? 'bg-cyan-500 text-slate-950'
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

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-4 text-left font-sans">Evaluation Metric</th>
                <th className="py-2.5 px-4 text-right text-emerald-300">Without PCA</th>
                <th className="py-2.5 px-4 text-right text-cyan-300">
                  With PCA (k = {selectedK})
                </th>
                <th className="py-2.5 px-4 text-right text-slate-400">Difference</th>
              </tr>
            </thead>
            <tbody>
              {performanceMatrixRows.map((row) => (
                <tr
                  key={row.metric}
                  className="border-b border-slate-800/60 hover:bg-slate-800/40"
                >
                  <td className="py-2.5 px-4 font-sans font-semibold text-slate-200">
                    {row.metric}
                  </td>
                  <td className="py-2.5 px-4 text-right text-emerald-300 font-bold">
                    {row.withoutPCA}
                  </td>
                  <td className="py-2.5 px-4 text-right text-cyan-300 font-bold">
                    {row.withPCA}
                  </td>
                  <td className="py-2.5 px-4 text-right text-slate-400">{row.diff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Part 2: 2x2 Regression Performance Comparison Bar Charts (Cell 19) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-5">
        <div className="flex items-center gap-2.5">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-lg font-bold text-white">
              2. Regression Performance Comparison (2×2 Metric Subplots)
            </h3>
            <p className="text-xs text-slate-400">
              Visualizing <span className="font-mono text-cyan-300">MAE</span>,{' '}
              <span className="font-mono text-cyan-300">MSE</span>,{' '}
              <span className="font-mono text-cyan-300">RMSE</span>, and{' '}
              <span className="font-mono text-cyan-300">R2 Score</span> side by side
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {metricSubplots.map((subplot) => (
            <div
              key={subplot.key}
              className="rounded-xl bg-slate-950/85 border border-slate-800 p-4 space-y-3"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-bold text-white">{subplot.title}</h4>
                  <p className="text-[11px] text-slate-400">{subplot.subtitle}</p>
                </div>
                <div className="text-right font-mono text-xs">
                  <span className="text-emerald-300 mr-2">
                    No PCA: {subplot.data[0].value}
                  </span>
                  <span className="text-cyan-300">PCA: {subplot.data[1].value}</span>
                </div>
              </div>

              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={subplot.data}
                    margin={{ top: 10, right: 15, left: -10, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '0.75rem',
                        fontSize: '12px',
                      }}
                    />
                    <Bar dataKey="value" name={subplot.title} radius={[6, 6, 0, 0]} barSize={56}>
                      {subplot.data.map((entry, idx) => (
                        <Cell key={`cell-${idx}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Part 3: Actual vs. Predicted Concrete Strength Scatter Plot (Cell 21) */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <GitCompare className="w-5 h-5 text-emerald-400" />
          <div>
            <h3 className="text-lg font-bold text-white">
              3. Regression Comparison: Actual vs. Predicted Concrete Strength
            </h3>
            <p className="text-xs text-slate-400">
              Comparing <span className="text-emerald-300 font-semibold">Without PCA</span> and{' '}
              <span className="text-cyan-300 font-semibold">With PCA (k = {selectedK})</span> against the dashed Perfect Prediction diagonal
            </p>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
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
                stroke="#f8fafc"
                strokeDasharray="5 5"
                label={{
                  value: 'Perfect Prediction',
                  fill: '#cbd5e1',
                  fontSize: 11,
                  position: 'insideTopLeft',
                }}
              />
              <Scatter
                name="Without PCA"
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

      {/* Part 4: Final Results Summary (Cell 23) */}
      <div className="rounded-2xl bg-slate-900/90 border border-cyan-500/35 p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <Award className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-lg font-bold text-white">
              4. Final Results Summary
            </h3>
            <p className="text-xs text-slate-400">
              Console output from the final cell of <span className="font-mono text-cyan-300">Concrete.ipynb</span>
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
          <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800">
            <div className="text-slate-400 text-[11px]">Original → Selected PCs</div>
            <div className="text-lg font-bold text-white mt-0.5">
              8 → {selectedK}
            </div>
            <div className="text-[11px] text-purple-300 mt-0.5">
              Dim Reduction: {modelB.dimensionalityReductionPercent.toFixed(1)}%
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800">
            <div className="text-slate-400 text-[11px]">Variance Retained</div>
            <div className="text-lg font-bold text-cyan-300 mt-0.5">
              {(modelB.retainedVarianceRatio * 100).toFixed(2)}%
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              Target: ≥95.0%
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800">
            <div className="text-slate-400 text-[11px]">RMSE (No PCA vs PCA)</div>
            <div className="text-lg font-bold text-white mt-0.5">
              {modelA.rmse.toFixed(4)} vs {modelB.rmse.toFixed(4)}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">Root Mean Squared Error</div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800">
            <div className="text-slate-400 text-[11px]">R² (No PCA vs PCA)</div>
            <div className="text-lg font-bold text-emerald-300 mt-0.5">
              {modelA.r2.toFixed(4)} vs <span className="text-cyan-300">{modelB.r2.toFixed(4)}</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              {modelB.r2 >= modelA.r2
                ? 'PCA maintained/improved accuracy'
                : 'PCA reduced regression accuracy'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
