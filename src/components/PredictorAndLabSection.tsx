import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  FlaskConical,
  RotateCcw,
  Play,
  GitCompareArrows,
  CheckCircle2,
} from 'lucide-react';
import {
  MLPipelineResult,
  predictConcreteStrength,
  CONCRETE_MIX_PRESETS,
} from '../ml/mlEngine';

interface PredictorAndLabSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
  isDetailedView?: boolean;
}

export const PredictorAndLabSection: React.FC<PredictorAndLabSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
  isDetailedView = false,
}) => {
  const [activeTab, setActiveTab] = useState<'predictor' | 'whatif'>('predictor');

  const defaultMeanInputs = useMemo(
    () =>
      pipeline.scaler.trainMeans.map((m, idx) =>
        Number(m.toFixed(idx === 7 ? 0 : 1))
      ),
    [pipeline]
  );

  // Predictor State
  const [inputs, setInputs] = useState<number[]>(defaultMeanInputs);
  const [committedInputs, setCommittedInputs] = useState<number[]>(defaultMeanInputs);

  // What-If Experiment State
  const [baselineMix] = useState<number[]>(CONCRETE_MIX_PRESETS[1].ingredients);
  const [modifiedMix, setModifiedMix] = useState<number[]>(CONCRETE_MIX_PRESETS[0].ingredients);

  const prediction = useMemo(
    () => predictConcreteStrength(pipeline, committedInputs, selectedK),
    [pipeline, committedInputs, selectedK]
  );

  const baselinePred = useMemo(
    () => predictConcreteStrength(pipeline, baselineMix, selectedK),
    [pipeline, baselineMix, selectedK]
  );

  const modifiedPred = useMemo(
    () => predictConcreteStrength(pipeline, modifiedMix, selectedK),
    [pipeline, modifiedMix, selectedK]
  );

  const wcRatio = committedInputs[0] > 0 ? committedInputs[3] / committedInputs[0] : 0;

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

  const whatIfChartData = [
    {
      model: 'Without PCA (8 Features)',
      Baseline: Number(baselinePred.predWithoutPCA.toFixed(2)),
      Modified: Number(modifiedPred.predWithoutPCA.toFixed(2)),
    },
    {
      model: `With PCA (k=${selectedK})`,
      Baseline: Number(baselinePred.predWithPCA.toFixed(2)),
      Modified: Number(modifiedPred.predWithPCA.toFixed(2)),
    },
  ];

  const getStrengthGrade = (mpa: number) => {
    if (mpa < 20) return { label: 'Low-Grade Mix', color: 'text-amber-400' };
    if (mpa < 40) return { label: 'Standard Structural Mix', color: 'text-cyan-400' };
    if (mpa < 60) return { label: 'High-Strength Concrete', color: 'text-emerald-400' };
    return { label: 'Ultra-High Performance', color: 'text-purple-400' };
  };

  const gradeNoPCA = getStrengthGrade(prediction.predWithoutPCA);
  const gradePCA = getStrengthGrade(prediction.predWithPCA);

  return (
    <div className="space-y-6">
      {/* Top Summary + Sub-Tab Switcher */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800/90 p-6 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <h2 className="text-xl font-bold text-white">
            Concrete Compressive Strength Predictor &amp; Mix Experiment Lab
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>
              Adjust the 8 ingredients to run real-time inference through your trained{' '}
              <strong className="text-white">StandardScaler → PCA (k={selectedK}) → OLS</strong>{' '}
              pipeline.
            </span>
          </p>
        </div>

        <div className="inline-flex rounded-xl bg-slate-950 border border-slate-800 p-1 self-start lg:self-center shrink-0">
          <button
            onClick={() => setActiveTab('predictor')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'predictor'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <FlaskConical className="w-4 h-4" />
            1. Strength Predictor
          </button>
          <button
            onClick={() => setActiveTab('whatif')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'whatif'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <GitCompareArrows className="w-4 h-4" />
            2. What-If Mix Lab
          </button>
        </div>
      </div>

      {/* TAB 1: Live Concrete Strength Predictor */}
      {(activeTab === 'predictor' || isDetailedView) && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 7 Cols: 8 Ingredient Sliders + Number Inputs */}
          <div className="lg:col-span-7 rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-white">8 Concrete Ingredient Controls</h3>
                <p className="text-xs text-slate-400">
                  Bounded to actual dataset min and max ranges
                </p>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {CONCRETE_MIX_PRESETS.slice(0, 3).map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => {
                      setInputs([...preset.ingredients]);
                      setCommittedInputs([...preset.ingredients]);
                    }}
                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700"
                  >
                    {preset.name}
                  </button>
                ))}
                <button
                  onClick={handleReset}
                  className="px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-slate-800 text-xs text-amber-300 border border-slate-800 flex items-center gap-1"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset Inputs
                </button>
              </div>
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
                      <span>Max: {stat.max}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center justify-between pt-2">
              <div className="text-xs font-mono text-slate-400">
                Water-to-Cement (W/C) Ratio:{' '}
                <strong className="text-cyan-300">{wcRatio.toFixed(2)}</strong>
              </div>
              <button
                onClick={() => setCommittedInputs([...inputs])}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-md shadow-cyan-500/15"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Predict Concrete Strength
              </button>
            </div>
          </div>

          {/* Right 5 Cols: Side-by-Side Predictions & Visual Gauges */}
          <div className="lg:col-span-5 rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 flex flex-col justify-between space-y-5">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white">Live Model Predictions</h3>
              <div className="flex items-center gap-1">
                <span className="text-xs text-slate-400 mr-1">PCs:</span>
                {[4, 6, 7, 8].map((k) => (
                  <button
                    key={k}
                    onClick={() => onSelectK(k)}
                    className={`px-2 py-0.5 rounded text-xs font-mono ${
                      selectedK === k
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    k={k}
                  </button>
                ))}
              </div>
            </div>

            {/* Prediction Without PCA */}
            <div className="p-5 rounded-xl bg-slate-950/90 border border-emerald-500/35 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-emerald-400 font-semibold uppercase tracking-wider">
                  Predicted Strength Without PCA
                </span>
                <span className="font-mono text-slate-400">8 Features</span>
              </div>
              <div className="text-3xl font-bold font-mono text-white">
                {prediction.predWithoutPCA.toFixed(2)}{' '}
                <span className="text-base font-normal text-slate-400">MPa</span>
              </div>
              <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-cyan-400 transition-all duration-300"
                  style={{
                    width: `${Math.min(
                      100,
                      Math.max(5, (prediction.predWithoutPCA / 85) * 100)
                    )}%`,
                  }}
                />
              </div>
              <div className={`text-xs font-medium ${gradeNoPCA.color}`}>{gradeNoPCA.label}</div>
            </div>

            {/* Prediction With PCA */}
            <div className="p-5 rounded-xl bg-slate-950/90 border border-cyan-500/35 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-cyan-400 font-semibold uppercase tracking-wider">
                  Predicted Strength With PCA
                </span>
                <span className="font-mono text-cyan-300">k = {selectedK} PCs</span>
              </div>
              <div className="text-3xl font-bold font-mono text-white">
                {prediction.predWithPCA.toFixed(2)}{' '}
                <span className="text-base font-normal text-slate-400">MPa</span>
              </div>
              <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-300"
                  style={{
                    width: `${Math.min(100, Math.max(5, (prediction.predWithPCA / 85) * 100))}%`,
                  }}
                />
              </div>
              <div className={`text-xs font-medium ${gradePCA.color}`}>{gradePCA.label}</div>
            </div>

            {/* Difference Card */}
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex items-center justify-between">
              <div>
                <div className="text-xs text-slate-400">
                  Prediction Difference (With PCA − Without PCA)
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  Absolute Deviation: {prediction.absDifference.toFixed(2)} MPa
                </div>
              </div>
              <div className="text-xl font-bold font-mono text-amber-300">
                {prediction.difference >= 0 ? '+' : ''}
                {prediction.difference.toFixed(2)} MPa
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Concrete Mix Experiment Lab (What-If Comparison) */}
      {(activeTab === 'whatif' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-bold text-white">
                Concrete Mix Experiment Lab (Baseline vs. Modified Mixture)
              </h3>
              <p className="text-xs text-slate-400">
                Modify Cement, Water, Superplasticizer, or Age below to see how both regression
                models respond.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Key Ingredient Modifiers */}
            <div className="space-y-3">
              <div className="text-xs font-semibold text-slate-300">
                Adjust Modified Mix Ingredients (Compared to Standard 28-Day Baseline):
              </div>
              {[0, 3, 4, 7].map((idx) => {
                const f = pipeline.featureMeta[idx];
                const stat = pipeline.columnStats[idx];
                const stepVal = idx === 7 ? 1 : 0.5;
                const delta = modifiedMix[idx] - baselineMix[idx];
                return (
                  <div
                    key={f.key}
                    className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-white">
                        {f.name} ({f.unit})
                      </span>
                      <span className="font-mono text-cyan-300">
                        Base: {baselineMix[idx]} → Mod: <strong>{modifiedMix[idx]}</strong> (
                        {delta >= 0 ? '+' : ''}
                        {delta.toFixed(1)})
                      </span>
                    </div>
                    <input
                      type="range"
                      min={stat.min}
                      max={stat.max}
                      step={stepVal}
                      value={modifiedMix[idx]}
                      onChange={(e) => {
                        const next = [...modifiedMix];
                        next[idx] = Number(e.target.value);
                        setModifiedMix(next);
                      }}
                      className="w-full accent-purple-400 cursor-pointer"
                    />
                  </div>
                );
              })}
            </div>

            {/* Bar Chart Comparing Baseline vs Modified Predictions */}
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                <div className="p-3.5 rounded-xl bg-slate-950 border border-emerald-500/30">
                  <div className="text-slate-400 font-sans">Without PCA Delta</div>
                  <div className="text-lg font-bold text-emerald-300 mt-1">
                    {modifiedPred.predWithoutPCA - baselinePred.predWithoutPCA >= 0 ? '+' : ''}
                    {(modifiedPred.predWithoutPCA - baselinePred.predWithoutPCA).toFixed(2)} MPa
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {baselinePred.predWithoutPCA.toFixed(1)} →{' '}
                    {modifiedPred.predWithoutPCA.toFixed(1)} MPa
                  </div>
                </div>

                <div className="p-3.5 rounded-xl bg-slate-950 border border-cyan-500/30">
                  <div className="text-slate-400 font-sans">With PCA (k={selectedK}) Delta</div>
                  <div className="text-lg font-bold text-cyan-300 mt-1">
                    {modifiedPred.predWithPCA - baselinePred.predWithPCA >= 0 ? '+' : ''}
                    {(modifiedPred.predWithPCA - baselinePred.predWithPCA).toFixed(2)} MPa
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {baselinePred.predWithPCA.toFixed(1)} → {modifiedPred.predWithPCA.toFixed(1)}{' '}
                    MPa
                  </div>
                </div>
              </div>

              <div className="h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={whatIfChartData}
                    margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="model" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
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
                      dataKey="Baseline"
                      name="Baseline Mix (MPa)"
                      fill="#64748b"
                      radius={[6, 6, 0, 0]}
                    />
                    <Bar
                      dataKey="Modified"
                      name="Modified Mix (MPa)"
                      fill="#a855f7"
                      radius={[6, 6, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
