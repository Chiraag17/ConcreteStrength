# Runtime & TypeScript Error Fix Plan — Concrete Strength AI Lab

## Root Cause Identified
During the UI simplification refactor, several component props referenced shorthand property names that did not match the exact data structures returned by `runConcreteMLPipeline` and `predictConcreteStrength` in `mlEngine.ts`, causing runtime `TypeError: Cannot read properties of undefined` crashes when rendering:

1. **Dataset & Overview Properties**:
   - `pipeline.totalRecords` → should be `pipeline.totalRows`
   - `f.label` / `f.shortLabel` → should be `f.name` / `f.shortName` (from `FeatureMeta` & `TARGET_METADATA`)
   - `pipeline.rawDistributions[i]` → should be `pipeline.columnStats[i].histogram`
2. **Normalization Properties**:
   - `pipeline.scaler.normalizedDistributions[i]` → should use `.bins` (`{ binCenter, label, count }[]`)
   - `pipeline.scaler.sampleComparisons` → should be `pipeline.scaler.sampleTransformations`
3. **PCA Simulator Properties**:
   - `dimensionalityReductionPct` → should be `dimensionalityReductionPercent`
   - `pipeline.pca.weightedContributionsByK` → should be `pipeline.pca.featureImportanceByK`
   - `pipeline.pca.projected2D` → should be `pipeline.pca.projectedSamples` (`pcCoords[0]`, `pcCoords[1]`, `actualStrength`)
4. **Regression & Predictor Properties**:
   - `model.testPoints` → should map `pipeline.regression.testActuals` with `model.testPredictions`
   - `CONCRETE_MIX_PRESETS[i].values` → should be `CONCRETE_MIX_PRESETS[i].ingredients`
   - `prediction.predictedWithoutPCA` / `predictedWithPCA` → should be `prediction.predWithoutPCA` / `predWithPCA`
5. **Strict TypeScript Check in `mlEngine.ts`**:
   - Explicitly type the identity matrix `V` in `eigenDecompositionSymmetric` as `number[][]` so `tsc --noEmit` passes with zero warnings or errors.

## Verification
- Run `lint_applet` (`tsc --noEmit`) to guarantee zero TypeScript or property mismatches across the entire codebase.
- Run `compile_applet` (`vite build`) and verify the dev server responds cleanly on port 3000.
