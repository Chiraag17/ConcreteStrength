import React, { useState, useEffect } from 'react';
import {
  Lightbulb,
  GraduationCap,
  Download,
  CheckCircle2,
  HelpCircle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface ModelInsightsSectionProps {
  pipeline: MLPipelineResult;
  selectedK: number;
  onSelectK: (k: number) => void;
  isDetailedView?: boolean;
}

export const ModelInsightsSection: React.FC<ModelInsightsSectionProps> = ({
  pipeline,
  selectedK,
  onSelectK,
  isDetailedView = false,
}) => {
  const [showVivaDrawer, setShowVivaDrawer] = useState<boolean>(isDetailedView);

  useEffect(() => {
    setShowVivaDrawer(isDetailedView);
  }, [isDetailedView]);

  const modelNoPCA = pipeline.regression.withoutPCA;
  const modelWithPCA = pipeline.regression.withPCAByK[selectedK];

  const retainedPct = modelWithPCA.retainedVarianceRatio * 100;
  const lostPct = modelWithPCA.lostVarianceRatio * 100;

  const handleExportJSON = () => {
    const report = {
      project: 'Concrete Strength AI Lab — PCA & Regression Summary',
      totalRecords: pipeline.totalRows,
      selectedPrincipalComponents: selectedK,
      retainedVariancePct: Number(retainedPct.toFixed(4)),
      lostVariancePct: Number(lostPct.toFixed(4)),
      regressionWithoutPCA: {
        r2: Number(modelNoPCA.r2.toFixed(6)),
        rmse: Number(modelNoPCA.rmse.toFixed(6)),
        mae: Number(modelNoPCA.mae.toFixed(6)),
      },
      regressionWithPCA: {
        k: selectedK,
        r2: Number(modelWithPCA.r2.toFixed(6)),
        rmse: Number(modelWithPCA.rmse.toFixed(6)),
        mae: Number(modelWithPCA.mae.toFixed(6)),
      },
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `concrete_pca_summary_k${selectedK}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const vivaQuestions = [
    {
      q: '1. Why did we normalize concrete ingredients with StandardScaler before PCA?',
      a: 'Coarse Aggregate (~973 kg/m³) and Cement (~283 kg/m³) have variances thousands of times larger than Superplasticizer (~6.2 kg/m³). Because PCA maximizes variance, unscaled PCA would be dominated by high-magnitude columns. StandardScaler sets every ingredient to mean = 0 and variance = 1.',
    },
    {
      q: '2. Does selecting 6 Principal Components mean we deleted 2 original ingredients?',
      a: 'No! PCA does not drop columns. Each Principal Component is a linear combination (weighted sum) of all 8 standardized concrete ingredients. Our weighted contribution analysis shows all 8 ingredients contribute ~11.5% to 12.5% each at k = 6.',
    },
    {
      q: '3. Why did Regression R² drop from 0.6275 (Without PCA) to 0.5679 (With PCA at k=6)?',
      a: 'PCA is an unsupervised technique—it finds directions of maximum variance in the inputs (X) without looking at the target (y). The discarded 2.69% variance in PC7 and PC8 still contains physical signal correlated with Compressive Strength.',
    },
    {
      q: '4. Which number of components gives the best balance?',
      a: 'For strict ≥95% variance compression, k = 6 retains 97.31% variance with 25% dimensionality reduction. For regression accuracy, k = 7 retains 99.61% variance and achieves R² = 0.6255 (nearly matching the 8-feature baseline R² = 0.6275).',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Executive Conclusions Card */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800/90 p-6 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <Lightbulb className="w-5 h-5 text-cyan-400" />
            <h2 className="text-xl font-bold text-white">
              Model Insights &amp; Final Project Conclusions
            </h2>
          </div>
          <button
            onClick={handleExportJSON}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-2 self-start sm:self-center"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            Export Summary JSON
          </button>
        </div>

        {/* 8 Required Summary Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <div className="text-xs text-slate-400">Original Features</div>
            <div className="text-xl font-bold font-mono text-white mt-1">8 Ingredients</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-cyan-500/30">
            <div className="text-xs text-slate-400">Selected PCs (k)</div>
            <div className="text-xl font-bold font-mono text-cyan-300 mt-1">{selectedK} of 8</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-emerald-500/30">
            <div className="text-xs text-slate-400">Variance Retained</div>
            <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
              {retainedPct.toFixed(2)}%
            </div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-rose-500/30">
            <div className="text-xs text-slate-400">Variance Lost</div>
            <div className="text-xl font-bold font-mono text-rose-300 mt-1">
              {lostPct.toFixed(2)}%
            </div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <div className="text-xs text-slate-400">R² Without PCA</div>
            <div className="text-xl font-bold font-mono text-white mt-1">
              {modelNoPCA.r2.toFixed(4)}
            </div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-cyan-500/30">
            <div className="text-xs text-slate-400">R² With PCA (k={selectedK})</div>
            <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
              {modelWithPCA.r2.toFixed(4)}
            </div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800">
            <div className="text-xs text-slate-400">RMSE Comparison</div>
            <div className="text-lg font-bold font-mono text-white mt-1">
              {modelWithPCA.rmse.toFixed(2)} vs {modelNoPCA.rmse.toFixed(2)}
            </div>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-purple-500/30">
            <div className="text-xs text-slate-400">Best Practical Balance</div>
            <div className="text-lg font-bold font-mono text-purple-300 mt-1">
              k=6 (Var) / k=7 (R²)
            </div>
          </div>
        </div>

        {/* Quick Component Switcher */}
        <div className="flex items-center gap-2 pt-1">
          <span className="text-xs text-slate-400">Inspect conclusions at k =</span>
          {[4, 6, 7, 8].map((k) => (
            <button
              key={k}
              onClick={() => onSelectK(k)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold ${
                selectedK === k
                  ? 'bg-cyan-500 text-slate-950'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              k={k}
            </button>
          ))}
        </div>
      </div>

      {/* Beginner-Friendly Explanation for Presentation */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
        <div className="flex items-center gap-2.5">
          <GraduationCap className="w-5 h-5 text-purple-400" />
          <h3 className="text-base font-bold text-white">
            3-Point Presentation Summary (Ready to Speak in Class)
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="text-xs font-bold uppercase tracking-wider text-cyan-400">
              1. What PCA Does Here
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Instead of dropping ingredients, PCA rotates the 8 standardized concrete ingredients
              into 8 uncorrelated recipes (Principal Components) ranked from highest variance (PC1:
              28.02%) to lowest (PC8: 0.39%).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              2. Retained Variance Result
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              With <strong>6 Principal Components</strong>, we retain <strong>97.31%</strong> of
              total dataset information while reducing dimensionality by <strong>25%</strong>, and
              all 8 original ingredients contribute almost equally (~12% each).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
            <div className="text-xs font-bold uppercase tracking-wider text-purple-400">
              3. Regression Impact Verdict
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Because Concrete Strength only has 8 features, compressing to 6 PCs trades{' '}
              <strong>5.96% R² accuracy</strong> (0.6275 → 0.5679) for a 25% smaller feature space,
              proving that low-variance directions (PC7) can still hold predictive value.
            </p>
          </div>
        </div>
      </div>

      {/* Expandable Drawer for Professor Viva Q&A Cards */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowVivaDrawer(!showVivaDrawer)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
        >
          <div className="flex items-center gap-2.5">
            <HelpCircle className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-slate-200">
              Common Professor Viva Questions &amp; Model Answers (4 Key Defenses)
            </span>
          </div>
          <span className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            {showVivaDrawer ? 'Hide Q&A' : 'Show Viva Q&A'}
            {showVivaDrawer ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </span>
        </button>

        {showVivaDrawer && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 grid grid-cols-1 md:grid-cols-2 gap-4">
            {vivaQuestions.map((item) => (
              <div
                key={item.q}
                className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2"
              >
                <div className="text-xs font-bold text-cyan-300 flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                  <span>{item.q}</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed pl-6">{item.a}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
