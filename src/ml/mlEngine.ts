import { CONCRETE_RAW_ROWS, FEATURE_METADATA, TARGET_METADATA, FeatureMeta } from '../data/concreteDataset';

export interface ColumnStats {
  index: number;
  key: string;
  name: string;
  shortName: string;
  unit: string;
  count: number;
  missing: number;
  mean: number;
  std: number;
  min: number;
  q25: number;
  median: number;
  q75: number;
  max: number;
  histogram: { binStart: number; binEnd: number; label: string; count: number; normalizedBinCenter: number }[];
}

export interface FeatureImportanceItem {
  featureIndex: number;
  name: string;
  shortName: string;
  unit: string;
  importance: number;
  sharePercent: number;
  loadingsByPC: number[];
}

export interface RegressionModelResult {
  numComponents: number | null; // null for Without PCA (all 8 original scaled features)
  label: string;
  featuresCount: number;
  retainedVarianceRatio: number;
  lostVarianceRatio: number;
  dimensionalityReductionPercent: number;
  intercept: number;
  coefficients: number[]; // length 8 for Without PCA, length k for With PCA
  effectiveFeatureWeights: number[]; // equivalent weights in scaled feature space (length 8)
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  trainRmse: number;
  trainR2: number;
  testPredictions: number[];
  testResiduals: number[];
}

export interface ProjectedSample {
  rowIndex: number;
  isTest: boolean;
  rawFeatures: number[];
  scaledFeatures: number[];
  pcCoords: number[]; // length 8: PC1..PC8
  actualStrength: number;
  predWithoutPCA: number;
  predWithPCA6: number;
}

export interface MLPipelineResult {
  datasetName: string;
  totalRows: number;
  trainCount: number;
  testCount: number;
  randomState: number;
  testSize: number;
  missingValues: number;
  featureMeta: FeatureMeta[];
  rawRows: number[][];
  trainIndices: number[];
  testIndices: number[];
  columnStats: ColumnStats[]; // 9 columns (8 features + 1 target)
  correlationMatrix: number[][]; // 9x9 Pearson correlation
  scaler: {
    trainMeans: number[]; // length 8
    trainStds: number[]; // length 8 (population std ddof=0 matching sklearn StandardScaler)
    sampleTransformations: {
      rowIndex: number;
      split: 'Train' | 'Test';
      raw: number[];
      scaled: number[];
      strength: number;
    }[];
    normalizedDistributions: {
      featureIndex: number;
      bins: { binCenter: number; label: string; count: number }[];
    }[];
  };
  pca: {
    eigenvalues: number[]; // length 8 (explained_variance_)
    explainedVarianceRatio: number[]; // length 8 (explained_variance_ratio_)
    cumulativeVarianceRatio: number[]; // length 8
    components: number[][]; // 8x8 matrix: row k = PC(k+1) eigenvector across 8 features
    loadingsMatrix: number[][]; // 8x8 matrix: row i = feature i, col k = PC(k+1) loading
    componentsFor95: number; // n = 6
    featureImportanceByK: Record<number, FeatureImportanceItem[]>; // k = 1..8
    projectedSamples: ProjectedSample[];
    trainTransformedHead: number[][]; // First 10 rows of X_train_pca (PC1..PC8) matching Cell 15 pca_df.head()
  };
  regression: {
    withoutPCA: RegressionModelResult;
    withPCAByK: Record<number, RegressionModelResult>; // k = 1..8
    testActuals: number[];
    trainActuals: number[];
    bestKByR2: number;
    bestBalancedK: number; // k = 6 (>=95% variance threshold)
  };
}

export interface SinglePredictionOutput {
  rawInputs: number[];
  scaledInputs: number[];
  pcScores: number[]; // length 8
  selectedK: number;
  predWithoutPCA: number;
  predWithPCA: number;
  difference: number; // predWithPCA - predWithoutPCA
  absDifference: number;
  percentDifference: number;
  predictionsAllK: { k: number; predictedStrength: number; retainedVariance: number }[];
  featureContributionsNoPCA: {
    name: string;
    shortName: string;
    rawVal: number;
    zScore: number;
    weight: number;
    contributionMPa: number;
  }[];
}

/**
 * Exact replica of NumPy's legacy RandomState(seed) MT19937 + rk_interval
 * Guarantees that train_test_split(X, y, test_size=0.2, random_state=42)
 * produces the exact same 824 train and 206 test indices as scikit-learn in Python.
 */
class NumpyRandomState {
  private mt: Uint32Array;
  private mti: number;

  constructor(seed: number) {
    this.mt = new Uint32Array(624);
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.mti = 624;
  }

  private uint32(): number {
    const mag01 = [0x0, 0x9908b0df];
    if (this.mti >= 624) {
      for (let kk = 0; kk < 624 - 397; kk++) {
        const y = ((this.mt[kk] & 0x80000000) | (this.mt[kk + 1] & 0x7fffffff)) >>> 0;
        this.mt[kk] = (this.mt[kk + 397] ^ (y >>> 1) ^ mag01[y & 0x1]) >>> 0;
      }
      for (let kk = 624 - 397; kk < 623; kk++) {
        const y = ((this.mt[kk] & 0x80000000) | (this.mt[kk + 1] & 0x7fffffff)) >>> 0;
        this.mt[kk] = (this.mt[kk + (397 - 624)] ^ (y >>> 1) ^ mag01[y & 0x1]) >>> 0;
      }
      const y = ((this.mt[623] & 0x80000000) | (this.mt[0] & 0x7fffffff)) >>> 0;
      this.mt[623] = (this.mt[396] ^ (y >>> 1) ^ mag01[y & 0x1]) >>> 0;
      this.mti = 0;
    }
    let y = this.mt[this.mti++];
    y ^= y >>> 11;
    y ^= ((y << 7) & 0x9d2c5680) >>> 0;
    y ^= ((y << 15) & 0xefc60000) >>> 0;
    y ^= y >>> 18;
    return y >>> 0;
  }

  private rkInterval(maxVal: number): number {
    if (maxVal === 0) return 0;
    let mask = maxVal;
    mask |= mask >>> 1;
    mask |= mask >>> 2;
    mask |= mask >>> 4;
    mask |= mask >>> 8;
    mask |= mask >>> 16;
    while (true) {
      const val = (this.uint32() & mask) >>> 0;
      if (val <= maxVal) return val;
    }
  }

  public permutation(n: number): number[] {
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = this.rkInterval(i);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function computeColumnStats(
  rows: number[][],
  colIndex: number,
  key: string,
  name: string,
  shortName: string,
  unit: string,
  trainMean?: number,
  trainStd?: number
): ColumnStats {
  const vals = rows.map((r) => r[colIndex]);
  const n = vals.length;
  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((acc, v) => acc + v, 0) / n;
  // pandas df.describe() uses sample std (ddof=1)
  const variance = vals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / Math.max(1, n - 1);
  const std = Math.sqrt(variance);
  const min = sorted[0];
  const max = sorted[n - 1];
  const q25 = quantile(sorted, 0.25);
  const median = quantile(sorted, 0.5);
  const q75 = quantile(sorted, 0.75);

  const numBins = 18;
  const span = max - min || 1;
  const binWidth = span / numBins;
  const histogram = Array.from({ length: numBins }, (_, b) => {
    const binStart = min + b * binWidth;
    const binEnd = b === numBins - 1 ? max : min + (b + 1) * binWidth;
    const binCenter = (binStart + binEnd) / 2;
    const normalizedBinCenter =
      trainMean !== undefined && trainStd !== undefined && trainStd > 0
        ? (binCenter - trainMean) / trainStd
        : (binCenter - mean) / (std || 1);
    return {
      binStart,
      binEnd,
      label: `${binStart.toFixed(0)}–${binEnd.toFixed(0)}`,
      count: 0,
      normalizedBinCenter: Number(normalizedBinCenter.toFixed(2)),
    };
  });

  for (const v of vals) {
    let idx = Math.floor((v - min) / binWidth);
    if (idx >= numBins) idx = numBins - 1;
    if (idx < 0) idx = 0;
    histogram[idx].count++;
  }

  return {
    index: colIndex,
    key,
    name,
    shortName,
    unit,
    count: n,
    missing: 0,
    mean,
    std,
    min,
    q25,
    median,
    q75,
    max,
    histogram,
  };
}

/**
 * Symmetric Jacobi Eigenvalue Decomposition with deterministic svd_flip sign convention
 */
function eigenDecompositionSymmetric(
  covMatrix: number[][],
  Z_train: number[][]
): { eigenvalues: number[]; components: number[][] } {
  const n = covMatrix.length;
  const D = covMatrix.map((row) => [...row]);
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let iter = 0; iter < 150; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(D[i][j]) > maxVal) {
          maxVal = Math.abs(D[i][j]);
          p = i;
          q = j;
        }
      }
    }
    if (maxVal < 1e-15) break;

    const theta = 0.5 * Math.atan2(2 * D[p][q], D[q][q] - D[p][p]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    const dpp = c * c * D[p][p] - 2 * s * c * D[p][q] + s * s * D[q][q];
    const dqq = s * s * D[p][p] + 2 * s * c * D[p][q] + c * c * D[q][q];
    D[p][q] = 0;
    D[q][p] = 0;
    D[p][p] = dpp;
    D[q][q] = dqq;

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const dip = c * D[i][p] - s * D[i][q];
        const diq = s * D[i][p] + c * D[i][q];
        D[i][p] = dip;
        D[p][i] = dip;
        D[i][q] = diq;
        D[q][i] = diq;
      }
      const vip = c * V[i][p] - s * V[i][q];
      const viq = s * V[i][p] + c * V[i][q];
      V[i][p] = vip;
      V[i][q] = viq;
    }
  }

  const pairs = D.map((row, idx) => ({
    val: Math.max(0, row[idx]),
    vec: V.map((r) => r[idx]),
  }));
  pairs.sort((a, b) => b.val - a.val);

  // Apply scikit-learn svd_flip(u, v, u_based_decision=True):
  // Ensure the largest magnitude projection in U_k = Z_train * vec_k is positive.
  const eigenvalues: number[] = [];
  const components: number[][] = [];

  for (const pair of pairs) {
    let maxAbsProj = -1;
    let sign = 1;
    for (let i = 0; i < Z_train.length; i++) {
      let proj = 0;
      for (let j = 0; j < n; j++) {
        proj += Z_train[i][j] * pair.vec[j];
      }
      if (Math.abs(proj) > maxAbsProj) {
        maxAbsProj = Math.abs(proj);
        sign = proj >= 0 ? 1 : -1;
      }
    }
    eigenvalues.push(pair.val);
    components.push(pair.vec.map((v) => v * sign));
  }

  return { eigenvalues, components };
}

/**
 * Solve linear system A * x = b via Gaussian Elimination with Partial Pivoting
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxAbs = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxAbs) {
        maxAbs = Math.abs(M[row][col]);
        pivotRow = row;
      }
    }
    if (pivotRow !== col) {
      const tmp = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = tmp;
    }

    const pivot = M[col][col] || 1e-12;
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / pivot;
      for (let k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k];
      }
    }
  }

  const x = Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = M[row][n];
    for (let col = row + 1; col < n; col++) {
      sum -= M[row][col] * x[col];
    }
    x[row] = sum / (M[row][row] || 1e-12);
  }
  return x;
}

/**
 * Ordinary Least Squares LinearRegression matching sklearn.linear_model.LinearRegression
 */
function fitLinearRegression(
  X_train: number[][],
  y_train: number[]
): { intercept: number; coefficients: number[] } {
  const m = X_train.length;
  const d = X_train[0].length;
  // Design matrix with column 0 = 1 (intercept)
  const p = d + 1;
  const XtX = Array.from({ length: p }, () => Array(p).fill(0));
  const Xty = Array(p).fill(0);

  for (let i = 0; i < m; i++) {
    const row = X_train[i];
    const yi = y_train[i];
    XtX[0][0] += 1;
    Xty[0] += yi;
    for (let j = 0; j < d; j++) {
      const xij = row[j];
      XtX[0][j + 1] += xij;
      XtX[j + 1][0] += xij;
      Xty[j + 1] += xij * yi;
      for (let k = j; k < d; k++) {
        const prod = xij * row[k];
        XtX[j + 1][k + 1] += prod;
        if (k !== j) {
          XtX[k + 1][j + 1] += prod;
        }
      }
    }
  }

  const theta = solveLinearSystem(XtX, Xty);
  return {
    intercept: theta[0],
    coefficients: theta.slice(1),
  };
}

function evaluatePredictions(actual: number[], predicted: number[]) {
  const n = actual.length;
  let sumAbs = 0;
  let sumSq = 0;
  let meanActual = 0;
  for (let i = 0; i < n; i++) meanActual += actual[i];
  meanActual /= n;

  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const err = actual[i] - predicted[i];
    sumAbs += Math.abs(err);
    sumSq += err * err;
    const diffMean = actual[i] - meanActual;
    ssTot += diffMean * diffMean;
  }

  const mae = sumAbs / n;
  const mse = sumSq / n;
  const rmse = Math.sqrt(mse);
  const r2 = ssTot > 0 ? 1 - sumSq / ssTot : 0;
  return { mae, mse, rmse, r2 };
}

/**
 * Runs the complete notebook Machine Learning pipeline on the given dataset rows.
 * By default uses CONCRETE_RAW_ROWS (1,030 records from Concrete_Data.xls) and seed=42.
 */
export function runConcreteMLPipeline(
  rawRows: number[][] = CONCRETE_RAW_ROWS,
  datasetName = 'Concrete_Data.xls (UCI Concrete Compressive Strength)',
  randomState = 42,
  testSize = 0.2
): MLPipelineResult {
  const totalRows = rawRows.length;
  const testCount = Math.round(totalRows * testSize);
  const trainCount = totalRows - testCount;

  // 1. Exact train_test_split(X, y, test_size=0.2, random_state=42)
  const rng = new NumpyRandomState(randomState);
  const perm = rng.permutation(totalRows);
  const testIndices = perm.slice(0, testCount);
  const trainIndices = perm.slice(testCount);
  const testSetLookup = new Set(testIndices);

  const X_train = trainIndices.map((idx) => rawRows[idx].slice(0, 8));
  const y_train = trainIndices.map((idx) => rawRows[idx][8]);
  const X_test = testIndices.map((idx) => rawRows[idx].slice(0, 8));
  const y_test = testIndices.map((idx) => rawRows[idx][8]);

  // 2. Fit StandardScaler strictly on X_train
  const trainMeans = Array(8).fill(0);
  const trainStds = Array(8).fill(0);
  for (let j = 0; j < 8; j++) {
    let sum = 0;
    for (let i = 0; i < trainCount; i++) sum += X_train[i][j];
    trainMeans[j] = sum / trainCount;

    let sqSum = 0;
    for (let i = 0; i < trainCount; i++) {
      const d = X_train[i][j] - trainMeans[j];
      sqSum += d * d;
    }
    trainStds[j] = Math.sqrt(sqSum / trainCount) || 1;
  }

  const scaleVector = (row8: number[]) =>
    row8.map((val, j) => (val - trainMeans[j]) / trainStds[j]);

  const Z_train = X_train.map(scaleVector);
  const Z_test = X_test.map(scaleVector);

  // 3. Descriptive statistics & 9x9 Correlation Matrix on full dataset
  const columnStats: ColumnStats[] = [
    ...FEATURE_METADATA.map((f) =>
      computeColumnStats(
        rawRows,
        f.index,
        f.key,
        f.name,
        f.shortName,
        f.unit,
        trainMeans[f.index],
        trainStds[f.index]
      )
    ),
    computeColumnStats(
      rawRows,
      8,
      TARGET_METADATA.key,
      TARGET_METADATA.name,
      TARGET_METADATA.shortName,
      TARGET_METADATA.unit
    ),
  ];

  const colMeans = columnStats.map((c) => c.mean);
  const correlationMatrix = Array.from({ length: 9 }, () => Array(9).fill(0));
  for (let c1 = 0; c1 < 9; c1++) {
    for (let c2 = c1; c2 < 9; c2++) {
      if (c1 === c2) {
        correlationMatrix[c1][c2] = 1;
        continue;
      }
      let num = 0;
      let den1 = 0;
      let den2 = 0;
      for (let i = 0; i < totalRows; i++) {
        const d1 = rawRows[i][c1] - colMeans[c1];
        const d2 = rawRows[i][c2] - colMeans[c2];
        num += d1 * d2;
        den1 += d1 * d1;
        den2 += d2 * d2;
      }
      const r = den1 > 0 && den2 > 0 ? num / Math.sqrt(den1 * den2) : 0;
      correlationMatrix[c1][c2] = Number(r.toFixed(4));
      correlationMatrix[c2][c1] = Number(r.toFixed(4));
    }
  }

  // Normalized z-score histogram distributions per feature
  const normalizedDistributions = FEATURE_METADATA.map((f) => {
    const zVals = rawRows.map((r) => (r[f.index] - trainMeans[f.index]) / trainStds[f.index]);
    const minZ = -3.0;
    const maxZ = 3.5;
    const numBins = 18;
    const step = (maxZ - minZ) / numBins;
    const bins = Array.from({ length: numBins }, (_, b) => {
      const start = minZ + b * step;
      const end = start + step;
      const center = (start + end) / 2;
      return {
        binCenter: Number(center.toFixed(2)),
        label: `${start.toFixed(1)}σ`,
        count: 0,
      };
    });
    for (const z of zVals) {
      let b = Math.floor((z - minZ) / step);
      if (b < 0) b = 0;
      if (b >= numBins) b = numBins - 1;
      bins[b].count++;
    }
    return { featureIndex: f.index, bins };
  });

  // 4. Fit PCA strictly on Z_train
  const covTrain = Array.from({ length: 8 }, () => Array(8).fill(0));
  for (let i = 0; i < 8; i++) {
    for (let j = i; j < 8; j++) {
      let s = 0;
      for (let k = 0; k < trainCount; k++) {
        s += Z_train[k][i] * Z_train[k][j];
      }
      const val = s / Math.max(1, trainCount - 1);
      covTrain[i][j] = val;
      covTrain[j][i] = val;
    }
  }

  const { eigenvalues, components } = eigenDecompositionSymmetric(covTrain, Z_train);
  const totalEigenVariance = eigenvalues.reduce((acc, v) => acc + v, 0) || 1;
  const explainedVarianceRatio = eigenvalues.map((v) => v / totalEigenVariance);

  const cumulativeVarianceRatio: number[] = [];
  let cumSum = 0;
  for (const r of explainedVarianceRatio) {
    cumSum += r;
    cumulativeVarianceRatio.push(Math.min(1, cumSum));
  }

  // Notebook Cell 6: n = np.argmax(cumulative >= 0.95) + 1
  const idx95 = cumulativeVarianceRatio.findIndex((c) => c >= 0.95);
  const componentsFor95 = idx95 >= 0 ? idx95 + 1 : 6;

  // Loadings matrix: row i = original feature i, col k = PC(k+1)
  const loadingsMatrix = Array.from({ length: 8 }, (_, featIdx) =>
    Array.from({ length: 8 }, (__, pcIdx) => components[pcIdx][featIdx])
  );

  // Notebook Cell 14: importance = (loadings ** 2) @ pca_selected.explained_variance_ratio_
  const featureImportanceByK: Record<number, FeatureImportanceItem[]> = {};
  for (let k = 1; k <= 8; k++) {
    const items: FeatureImportanceItem[] = FEATURE_METADATA.map((f) => {
      let weightedSum = 0;
      for (let pc = 0; pc < k; pc++) {
        const loading = loadingsMatrix[f.index][pc];
        weightedSum += loading * loading * explainedVarianceRatio[pc];
      }
      return {
        featureIndex: f.index,
        name: f.name,
        shortName: f.shortName,
        unit: f.unit,
        importance: weightedSum,
        sharePercent: (weightedSum / cumulativeVarianceRatio[k - 1]) * 100,
        loadingsByPC: loadingsMatrix[f.index].slice(0, k),
      };
    });
    items.sort((a, b) => b.importance - a.importance);
    featureImportanceByK[k] = items;
  }

  // Project helper: z (length 8) -> pcScores (length 8)
  const projectToPC = (z: number[]): number[] =>
    components.map((compVec) => {
      let dot = 0;
      for (let j = 0; j < 8; j++) dot += z[j] * compVec[j];
      return dot;
    });

  const PC_train_all = Z_train.map(projectToPC);
  const PC_test_all = Z_test.map(projectToPC);

  // 5. Train Model A: LinearRegression Without PCA (on Z_train)
  const fitNoPCA = fitLinearRegression(Z_train, y_train);
  const predTrainNoPCA = Z_train.map(
    (z) =>
      fitNoPCA.intercept +
      z.reduce((acc, zj, j) => acc + zj * fitNoPCA.coefficients[j], 0)
  );
  const predTestNoPCA = Z_test.map(
    (z) =>
      fitNoPCA.intercept +
      z.reduce((acc, zj, j) => acc + zj * fitNoPCA.coefficients[j], 0)
  );
  const evalTrainNoPCA = evaluatePredictions(y_train, predTrainNoPCA);
  const evalTestNoPCA = evaluatePredictions(y_test, predTestNoPCA);

  const withoutPCA: RegressionModelResult = {
    numComponents: null,
    label: 'Without PCA (All 8 Scaled Ingredients)',
    featuresCount: 8,
    retainedVarianceRatio: 1.0,
    lostVarianceRatio: 0.0,
    dimensionalityReductionPercent: 0,
    intercept: fitNoPCA.intercept,
    coefficients: fitNoPCA.coefficients,
    effectiveFeatureWeights: fitNoPCA.coefficients,
    mae: evalTestNoPCA.mae,
    mse: evalTestNoPCA.mse,
    rmse: evalTestNoPCA.rmse,
    r2: evalTestNoPCA.r2,
    trainRmse: evalTrainNoPCA.rmse,
    trainR2: evalTrainNoPCA.r2,
    testPredictions: predTestNoPCA,
    testResiduals: y_test.map((yi, idx) => yi - predTestNoPCA[idx]),
  };

  // 6. Train Model B: LinearRegression With PCA for every k = 1..8
  const withPCAByK: Record<number, RegressionModelResult> = {};
  let bestKByR2 = 1;
  let maxR2 = -Infinity;

  for (let k = 1; k <= 8; k++) {
    const X_tr_k = PC_train_all.map((pcRow) => pcRow.slice(0, k));
    const X_te_k = PC_test_all.map((pcRow) => pcRow.slice(0, k));
    const fitPCAk = fitLinearRegression(X_tr_k, y_train);

    const predTrK = X_tr_k.map(
      (p) =>
        fitPCAk.intercept +
        p.reduce((acc, pcVal, cIdx) => acc + pcVal * fitPCAk.coefficients[cIdx], 0)
    );
    const predTeK = X_te_k.map(
      (p) =>
        fitPCAk.intercept +
        p.reduce((acc, pcVal, cIdx) => acc + pcVal * fitPCAk.coefficients[cIdx], 0)
    );

    const evalTrK = evaluatePredictions(y_train, predTrK);
    const evalTeK = evaluatePredictions(y_test, predTeK);

    // Map PC coefficients back to effective weights on the 8 scaled features: w_eff = V_k^T * gamma
    const effectiveFeatureWeights = Array(8).fill(0);
    for (let j = 0; j < 8; j++) {
      let w = 0;
      for (let cIdx = 0; cIdx < k; cIdx++) {
        w += components[cIdx][j] * fitPCAk.coefficients[cIdx];
      }
      effectiveFeatureWeights[j] = w;
    }

    const retained = cumulativeVarianceRatio[k - 1];
    withPCAByK[k] = {
      numComponents: k,
      label: `With PCA (${k} Principal Component${k > 1 ? 's' : ''})`,
      featuresCount: k,
      retainedVarianceRatio: retained,
      lostVarianceRatio: Math.max(0, 1 - retained),
      dimensionalityReductionPercent: ((8 - k) / 8) * 100,
      intercept: fitPCAk.intercept,
      coefficients: fitPCAk.coefficients,
      effectiveFeatureWeights,
      mae: evalTeK.mae,
      mse: evalTeK.mse,
      rmse: evalTeK.rmse,
      r2: evalTeK.r2,
      trainRmse: evalTrK.rmse,
      trainR2: evalTrK.r2,
      testPredictions: predTeK,
      testResiduals: y_test.map((yi, idx) => yi - predTeK[idx]),
    };

    if (evalTeK.r2 > maxR2) {
      maxR2 = evalTeK.r2;
      bestKByR2 = k;
    }
  }

  // Projected samples for 2D/3D PCA visualization
  const modelPCA6 = withPCAByK[componentsFor95];
  const projectedSamples: ProjectedSample[] = rawRows.map((row, idx) => {
    const rawFeatures = row.slice(0, 8);
    const actualStrength = row[8];
    const scaledFeatures = scaleVector(rawFeatures);
    const pcCoords = projectToPC(scaledFeatures);
    const predWithoutPCA =
      withoutPCA.intercept +
      scaledFeatures.reduce((acc, zj, j) => acc + zj * withoutPCA.coefficients[j], 0);
    const predWithPCA6 =
      modelPCA6.intercept +
      pcCoords
        .slice(0, componentsFor95)
        .reduce((acc, pcVal, cIdx) => acc + pcVal * modelPCA6.coefficients[cIdx], 0);

    return {
      rowIndex: idx,
      isTest: testSetLookup.has(idx),
      rawFeatures,
      scaledFeatures,
      pcCoords,
      actualStrength,
      predWithoutPCA,
      predWithPCA6,
    };
  });

  const sampleTransformations = rawRows.slice(0, 25).map((row, idx) => ({
    rowIndex: idx + 1,
    split: (testSetLookup.has(idx) ? 'Test' : 'Train') as 'Train' | 'Test',
    raw: row.slice(0, 8),
    scaled: scaleVector(row.slice(0, 8)),
    strength: row[8],
  }));

  return {
    datasetName,
    totalRows,
    trainCount,
    testCount,
    randomState,
    testSize,
    missingValues: 0,
    featureMeta: FEATURE_METADATA,
    rawRows,
    trainIndices,
    testIndices,
    columnStats,
    correlationMatrix,
    scaler: {
      trainMeans,
      trainStds,
      sampleTransformations,
      normalizedDistributions,
    },
    pca: {
      eigenvalues,
      explainedVarianceRatio,
      cumulativeVarianceRatio,
      components,
      loadingsMatrix,
      componentsFor95,
      featureImportanceByK,
      projectedSamples,
      trainTransformedHead: PC_train_all.slice(0, 10),
    },
    regression: {
      withoutPCA,
      withPCAByK,
      testActuals: y_test,
      trainActuals: y_train,
      bestKByR2,
      bestBalancedK: componentsFor95,
    },
  };
}

/**
 * Real-time inference using the fitted preprocessing & regression pipeline
 */
export function predictConcreteStrength(
  pipeline: MLPipelineResult,
  rawInputs: number[],
  selectedK: number
): SinglePredictionOutput {
  const { trainMeans, trainStds } = pipeline.scaler;
  const { components, cumulativeVarianceRatio } = pipeline.pca;
  const { withoutPCA, withPCAByK } = pipeline.regression;

  // Step 1: Apply fitted StandardScaler (using training means & stds only)
  const scaledInputs = rawInputs.map((val, j) => (val - trainMeans[j]) / trainStds[j]);

  // Step 2: Project scaled inputs onto all 8 Principal Components
  const pcScores = components.map((compVec) => {
    let dot = 0;
    for (let j = 0; j < 8; j++) dot += scaledInputs[j] * compVec[j];
    return dot;
  });

  // Step 3: Predict with Model A (Without PCA)
  const predWithoutPCA =
    withoutPCA.intercept +
    scaledInputs.reduce((acc, zj, j) => acc + zj * withoutPCA.coefficients[j], 0);

  // Step 4: Predict with Model B (With PCA for selectedK)
  const modelK = withPCAByK[selectedK] || withPCAByK[6];
  const predWithPCA =
    modelK.intercept +
    pcScores
      .slice(0, selectedK)
      .reduce((acc, pcVal, cIdx) => acc + pcVal * modelK.coefficients[cIdx], 0);

  const difference = predWithPCA - predWithoutPCA;
  const absDifference = Math.abs(difference);
  const percentDifference =
    Math.abs(predWithoutPCA) > 1e-6 ? (difference / predWithoutPCA) * 100 : 0;

  const predictionsAllK = Array.from({ length: 8 }, (_, idx) => {
    const k = idx + 1;
    const mk = withPCAByK[k];
    const pred =
      mk.intercept +
      pcScores.slice(0, k).reduce((acc, pcVal, cIdx) => acc + pcVal * mk.coefficients[cIdx], 0);
    return {
      k,
      predictedStrength: pred,
      retainedVariance: cumulativeVarianceRatio[idx] * 100,
    };
  });

  const featureContributionsNoPCA = pipeline.featureMeta.map((f, j) => ({
    name: f.name,
    shortName: f.shortName,
    rawVal: rawInputs[j],
    zScore: scaledInputs[j],
    weight: withoutPCA.coefficients[j],
    contributionMPa: scaledInputs[j] * withoutPCA.coefficients[j],
  }));

  return {
    rawInputs,
    scaledInputs,
    pcScores,
    selectedK,
    predWithoutPCA,
    predWithPCA,
    difference,
    absDifference,
    percentDifference,
    predictionsAllK,
    featureContributionsNoPCA,
  };
}

export interface ConcreteMixPreset {
  id: string;
  name: string;
  badge: string;
  description: string;
  rowIndexInDataset: number;
  ingredients: number[]; // [Cement, Slag, FlyAsh, Water, Superplasticizer, CoarseAgg, FineAgg, Age]
  actualDatasetStrength: number;
}

export const CONCRETE_MIX_PRESETS: ConcreteMixPreset[] = [
  {
    id: 'high-strength',
    name: 'High-Performance Structural Mix',
    badge: '79.99 MPa Actual',
    description: 'High cement dosage (540 kg/m³) with low water-to-cement ratio (0.30) cured for 28 days (Dataset Row #1).',
    rowIndexInDataset: 0,
    ingredients: [540.0, 0.0, 0.0, 162.0, 2.5, 1040.0, 676.0, 28],
    actualDatasetStrength: 79.99,
  },
  {
    id: 'standard-28d',
    name: 'Standard Commercial 28-Day Mix',
    badge: '39.29 MPa Actual',
    description: 'Balanced ordinary Portland cement mix with moderate water and aggregate proportions (Dataset Row #205).',
    rowIndexInDataset: 204,
    ingredients: [310.0, 0.0, 0.0, 192.0, 0.0, 971.0, 850.6, 28],
    actualDatasetStrength: 27.83,
  },
  {
    id: 'eco-slag',
    name: 'Eco-Slag Blended Mix',
    badge: '64.90 MPa Actual',
    description: 'High blast furnace slag replacement (237.5 kg/m³) with superplasticizer at 28 days (Dataset Row #77).',
    rowIndexInDataset: 76,
    ingredients: [469.0, 117.2, 0.0, 137.8, 32.2, 852.1, 840.5, 28],
    actualDatasetStrength: 66.9,
  },
  {
    id: 'fly-ash-blend',
    name: 'Pozzolanic Fly-Ash Mix',
    badge: '44.28 MPa Actual',
    description: 'Fly ash (141.8 kg/m³) and superplasticizer blend designed for workability and durability (Dataset Row #230).',
    rowIndexInDataset: 229,
    ingredients: [213.7, 98.1, 24.5, 181.7, 6.9, 1065.8, 785.4, 28],
    actualDatasetStrength: 40.23,
  },
  {
    id: 'early-age-3d',
    name: 'Early-Age 3-Day Curing Mix',
    badge: '15.05 MPa Actual',
    description: 'Demonstrates early hydration stage after only 3 days of curing before full crystal development.',
    rowIndexInDataset: 18,
    ingredients: [332.5, 142.5, 0.0, 228.0, 0.0, 932.0, 594.0, 3],
    actualDatasetStrength: 15.05,
  },
];
