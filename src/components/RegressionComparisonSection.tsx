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
  LineChart,
  Line,
  Legend,
} from 'recharts';
import { GitCompare, CheckCircle2 } from 'lucide-react';
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

  const kSeriesData = useMemo(() => {
    return [1, 2, 3, 4, 5, 6, 7, 8].map((k) => {
      const m = pipeline.regression.withPCAByK[k];
      return {
        kLabel: `${k} PC${k > 1 ? 's' : ''}`,
        k,
        r2WithPCA: Number(m.r2.toFixed(4)),
        rmseWithPCA: Number(m.rmse.toFixed(2)),
        retainedPct: Number((m.retainedVarianceRatio * 100).toFixed(1)),
      };
    });
  }, [pipeline.regression]);

  const scatterDataA = useMemo(
    () =>
      pipeline.regression.testActuals.map((actual, idx) => ({
        actual: Number(actual.toFixed(1)),
        predicted: Number((modelA.testPredictions[idx] ?? 0).toFixed(1)),
      })),
    [pipeline.regression.testActuals, modelA.testPredictions]
  );

  const scatterDataB = useMemo(
    () =>
      pipeline.regression.testActuals.map((actual, idx) => ({
        actual: Number(actual.toFixed(1)),
        predicted: Number((modelB.testPredictions[idx] ?? 0).toFixed(1)),
      })),
    [pipeline.regression.testActuals, modelB.testPredictions]
  );

  return (
    <div className="space-y-6">
      {/* Side-by-Side Comparison Header Card */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800 p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <GitCompare className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-xl font-bold text-white">
                Regression Without PCA vs. With PCA
              </h2>
              <p className="text-xs text-slate-400">
                Comparing Linear Regression accuracy on the 206 unseen test samples
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

        {/* Two Clean Scoreboard Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl bg-slate-950/90 border border-emerald-500/35 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
                Model A: Without PCA (All 8 Features)
              </span>
              <span className="text-xs font-mono text-slate-400">100% Variance</span>
            </div>
            <div className="grid grid-cols-4 gap-2 font-mono pt-1">
              <div>
                <div className="text-[11px] text-slate-500">R² Score</div>
                <div className="text-xl font-bold text-emerald-300">{modelA.r2.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">RMSE</div>
                <div className="text-xl font-bold text-white">{modelA.rmse.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MAE</div>
                <div className="text-xl font-bold text-slate-200">{modelA.mae.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MSE</div>
                <div className="text-xl font-bold text-slate-300">{modelA.mse.toFixed(1)}</div>
              </div>
            </div>
          </div>

          <div className="rounded-xl bg-slate-950/90 border border-cyan-500/35 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Model B: With PCA (k = {selectedK} Components)
              </span>
              <span className="text-xs font-mono text-cyan-300">
                {(modelB.retainedVarianceRatio * 100).toFixed(2)}% Variance
              </span>
            </div>
            <div className="grid grid-cols-4 gap-2 font-mono pt-1">
              <div>
                <div className="text-[11px] text-slate-500">R² Score</div>
                <div className="text-xl font-bold text-cyan-300">{modelB.r2.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">RMSE</div>
                <div className="text-xl font-bold text-white">{modelB.rmse.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MAE</div>
                <div className="text-xl font-bold text-slate-200">{modelB.mae.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-500">MSE</div>
                <div className="text-xl font-bold text-slate-300">{modelB.mse.toFixed(1)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Simple Takeaway Box */}
        <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs sm:text-sm text-slate-300 flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <strong className="text-white">Why does PCA at k = 6 have a slightly lower R² (0.5679 vs 0.6275)?</strong>{' '}
            PCA only looks at ingredient variance, not the target strength. Even though PC7 and PC8
            hold only <strong>2.69%</strong> of variance, they still carry useful information for
            predicting strength. Selecting <strong>k = 7</strong> recovers almost full accuracy (
            <strong className="text-emerald-300 font-mono">R² = 0.6255</strong>).
          </div>
        </div>
      </div>

      {/* Charts: R² vs PCs & RMSE vs PCs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              1. R² Score vs. Number of Principal Components
            </h3>
            <p className="text-xs text-slate-400">
              Higher is better. Green dashed line = Without PCA (0.6275)
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={kSeriesData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="kLabel" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
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
                    value: 'Without PCA (0.6275)',
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

        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">
              2. Prediction Error (RMSE) vs. Principal Components
            </h3>
            <p className="text-xs text-slate-400">
              Lower error is better. Green dashed line = Without PCA (9.80 MPa)
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={kSeriesData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="kLabel" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
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
                  y={Number(modelA.rmse.toFixed(2))}
                  stroke="#10b981"
                  strokeDasharray="5 5"
                  label={{
                    value: 'Without PCA (9.80 MPa)',
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

      {/* Actual vs Predicted Scatterplot */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div>
          <h3 className="text-base font-bold text-white">
            3. Actual vs. Predicted Concrete Strength (Test Set)
          </h3>
          <p className="text-xs text-slate-400">
            Points closer to the dashed diagonal line indicate more accurate predictions
          </p>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                type="number"
                dataKey="actual"
                name="Actual"
                domain={[0, 85]}
                unit=" MPa"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
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
                name="Without PCA (8 Features)"
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
    </div>
  );
};
