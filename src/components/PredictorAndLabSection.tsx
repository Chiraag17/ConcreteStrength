import React, { useState, useMemo } from 'react';
import { FlaskConical, RotateCcw, Play, Sigma } from 'lucide-react';
import { MLPipelineResult, predictConcreteStrength } from '../ml/mlEngine';

interface PredictorAndLabSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
}

export const PredictorAndLabSection: React.FC<PredictorAndLabSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
}) => {
  const defaultMeanInputs = useMemo(
    () =>
      pipeline.scaler.trainMeans.map((m, idx) =>
        Number(m.toFixed(idx === 7 ? 0 : 1))
      ),
    [pipeline]
  );

  const [inputs, setInputs] = useState<number[]>(defaultMeanInputs);
  const [committedInputs, setCommittedInputs] = useState<number[]>(defaultMeanInputs);

  const prediction = useMemo(
    () => predictConcreteStrength(pipeline, committedInputs, selectedK),
    [pipeline, committedInputs, selectedK]
  );

  const handleInputChange = (idx: number, val: number) => {
    const stat = pipeline.columnStats[idx];
    const clamped = Math.max(stat.min, Math.min(stat.max, val));
    const next = [...inputs];
    next[idx] = clamped;
    setInputs(next);
    setCommittedInputs(next);
  };

  const handleReset = () => {
    setInputs(defaultMeanInputs);
    setCommittedInputs(defaultMeanInputs);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: 8 Ingredient Sliders */}
        <div className="lg:col-span-7 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <FlaskConical className="w-5 h-5 text-cyan-400" />
              <div>
                <h2 className="text-lg font-bold text-white">
                  Adjust Concrete Mix Ingredients
                </h2>
                <p className="text-xs text-slate-400">
                  Use the sliders or number boxes to test any concrete mix
                </p>
              </div>
            </div>

            <button
              onClick={handleReset}
              className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-amber-300 border border-slate-700 flex items-center gap-1.5"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Inputs
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {pipeline.featureMeta.map((f, idx) => {
              const stat = pipeline.columnStats[idx];
              const stepVal = idx === 7 ? 1 : 0.5;
              return (
                <div
                  key={f.key}
                  className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <label className="text-xs font-semibold text-slate-200">
                      {f.shortName}{' '}
                      <span className="text-slate-500 font-normal">({f.unit})</span>
                    </label>
                    <input
                      type="number"
                      min={stat.min}
                      max={stat.max}
                      step={stepVal}
                      value={inputs[idx]}
                      onChange={(e) => handleInputChange(idx, Number(e.target.value))}
                      className="w-20 px-2 py-1 rounded-lg bg-slate-900 border border-slate-700 text-right font-mono text-xs text-cyan-300 focus:outline-none focus:border-cyan-400"
                    />
                  </div>
                  <input
                    type="range"
                    min={stat.min}
                    max={stat.max}
                    step={stepVal}
                    value={inputs[idx]}
                    onChange={(e) => handleInputChange(idx, Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>Min: {stat.min}</span>
                    <span>Z = {prediction.scaledInputs[idx].toFixed(2)}σ</span>
                    <span>Max: {stat.max}</span>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={() => setCommittedInputs([...inputs])}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-md shadow-cyan-500/15"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Predict Concrete Strength
            </button>
          </div>
        </div>

        {/* Right 5 Cols: Side-by-Side Predictions */}
        <div className="lg:col-span-5 rounded-2xl bg-slate-900/80 border border-slate-800 p-6 flex flex-col justify-between space-y-5">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-bold text-white">Predicted Strength Results</h3>
            <div className="flex items-center gap-1">
              <span className="text-xs text-slate-400 mr-1">PCs (k):</span>
              {[4, 6, 7, 8].map((k) => (
                <button
                  key={k}
                  onClick={() => onSelectK(k)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold ${
                    selectedK === k
                      ? 'bg-cyan-500 text-slate-950'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>

          {/* Without PCA Card */}
          <div className="p-5 rounded-xl bg-slate-950/90 border border-emerald-500/35 space-y-2.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-emerald-400 font-bold uppercase tracking-wider">
                Predicted Strength Without PCA
              </span>
              <span className="font-mono text-slate-400">All 8 Features</span>
            </div>
            <div className="text-3xl font-bold font-mono text-white">
              {prediction.predWithoutPCA.toFixed(2)}{' '}
              <span className="text-base font-normal text-slate-400">MPa</span>
            </div>
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-300"
                style={{
                  width: `${Math.min(100, Math.max(5, (prediction.predWithoutPCA / 85) * 100))}%`,
                }}
              />
            </div>
          </div>

          {/* With PCA Card */}
          <div className="p-5 rounded-xl bg-slate-950/90 border border-cyan-500/35 space-y-2.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-cyan-400 font-bold uppercase tracking-wider">
                Predicted Strength With PCA
              </span>
              <span className="font-mono text-cyan-300">k = {selectedK} PCs</span>
            </div>
            <div className="text-3xl font-bold font-mono text-white">
              {prediction.predWithPCA.toFixed(2)}{' '}
              <span className="text-base font-normal text-slate-400">MPa</span>
            </div>
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-300"
                style={{
                  width: `${Math.min(100, Math.max(5, (prediction.predWithPCA / 85) * 100))}%`,
                }}
              />
            </div>
          </div>

          {/* Difference Card */}
          <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-xs font-semibold text-slate-300">Difference Between Models</div>
              <div className="text-xs text-slate-500 mt-0.5">
                With PCA ({selectedK} PCs) vs. Without PCA
              </div>
            </div>
            <div className="text-2xl font-bold font-mono text-amber-300">
              {prediction.difference >= 0 ? '+' : ''}
              {prediction.difference.toFixed(2)} MPa
            </div>
          </div>
        </div>
      </div>

      {/* Live Projected Principal Component Values for Current Mix */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <Sigma className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-base font-bold text-white">
              Live PCA Transformed Coordinates for Current Mix (PC1 – PC{selectedK})
            </h3>
            <p className="text-xs text-slate-400">
              Computed by applying <span className="font-mono text-cyan-300">StandardScaler</span> and the{' '}
              <span className="font-mono text-cyan-300">PCA Mathematical Equations</span> to your selected mix
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono">
          {prediction.pcScores.slice(0, selectedK).map((score, idx) => (
            <div
              key={idx}
              className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800 text-center"
            >
              <div className="text-xs text-cyan-400 font-bold">PC{idx + 1}</div>
              <div className="text-base font-bold text-white mt-1">
                {score >= 0 ? '+' : ''}
                {score.toFixed(4)}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                Var: {(pipeline.pca.explainedVarianceRatio[idx] * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
