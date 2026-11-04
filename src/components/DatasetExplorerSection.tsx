import React, { useState, useMemo, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Search,
  ChevronLeft,
  ChevronRight,
  Table,
  BarChart3,
  Grid,
  Upload,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { MLPipelineResult } from '../ml/mlEngine';

interface DatasetExplorerSectionProps {
  pipeline: MLPipelineResult;
  onOpenUploadModal: () => void;
  isDetailedView?: boolean;
}

export const DatasetExplorerSection: React.FC<DatasetExplorerSectionProps> = ({
  pipeline,
  onOpenUploadModal,
  isDetailedView = false,
}) => {
  const [activeTab, setActiveTab] = useState<'distributions' | 'correlation' | 'table'>(
    'distributions'
  );
  const [selectedColIdx, setSelectedColIdx] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [page, setPage] = useState<number>(0);
  const [showStatsDrawer, setShowStatsDrawer] = useState<boolean>(isDetailedView);

  useEffect(() => {
    setShowStatsDrawer(isDetailedView);
  }, [isDetailedView]);

  const rowsPerPage = 12;
  const allColumns = pipeline.columnStats;
  const selectedStats = allColumns[selectedColIdx] || allColumns[0];
  const selectedDist = selectedStats ? selectedStats.histogram : [];

  const filteredRows = useMemo(() => {
    if (!searchQuery.trim()) return pipeline.rawRows.map((r, i) => ({ idx: i + 1, values: r }));
    const q = searchQuery.trim().toLowerCase();
    return pipeline.rawRows
      .map((r, i) => ({ idx: i + 1, values: r }))
      .filter(
        (item) =>
          String(item.idx).includes(q) || item.values.some((v) => v.toFixed(1).includes(q))
      );
  }, [pipeline.rawRows, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / rowsPerPage));
  const paginatedRows = filteredRows.slice(page * rowsPerPage, (page + 1) * rowsPerPage);

  const getCorrelationStyle = (r: number) => {
    const abs = Math.min(1, Math.abs(r));
    if (r >= 0) {
      return {
        backgroundColor: `rgba(6, 182, 212, ${0.1 + abs * 0.65})`,
        color: abs > 0.45 ? '#ffffff' : '#cbd5e1',
      };
    }
    return {
      backgroundColor: `rgba(244, 63, 94, ${0.1 + abs * 0.65})`,
      color: abs > 0.45 ? '#ffffff' : '#cbd5e1',
    };
  };

  return (
    <div className="space-y-6">
      {/* Top Summary Bar + 3-Bullet Takeaway */}
      <div className="rounded-2xl bg-slate-900/85 border border-slate-800/90 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-2">
            <h2 className="text-xl font-bold text-white">
              Concrete Compressive Strength Dataset ({pipeline.totalRows.toLocaleString()} Mixes)
            </h2>
            <ul className="space-y-1.5 text-xs sm:text-sm text-slate-300">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />
                <span>
                  <strong className="text-white">8 Input Ingredients</strong> (kg/m³ &amp; days)
                  predict{' '}
                  <strong className="text-cyan-300">
                    Compressive Strength (2.33 to 82.60 MPa)
                  </strong>{' '}
                  with <strong className="text-emerald-400">0 missing values</strong>.
                </span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>
                  <strong className="text-white">Top Positive Drivers:</strong> Cement (
                  <span className="font-mono text-cyan-300">r = +0.50</span>), Superplasticizer (
                  <span className="font-mono text-cyan-300">r = +0.37</span>), and Age (
                  <span className="font-mono text-cyan-300">r = +0.33</span>).
                </span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-rose-400 shrink-0" />
                <span>
                  <strong className="text-white">Top Negative Driver:</strong> Water (
                  <span className="font-mono text-rose-300">r = -0.29</span>) reduces strength and
                  strongly opposes Superplasticizer (
                  <span className="font-mono text-rose-300">r = -0.66</span>).
                </span>
              </li>
            </ul>
          </div>

          <button
            onClick={onOpenUploadModal}
            className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold text-slate-200 flex items-center gap-2 self-start lg:self-center shrink-0"
          >
            <Upload className="w-4 h-4 text-cyan-400" />
            Upload Custom CSV / Excel
          </button>
        </div>

        {/* 4 Compact Stat Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 pt-5 border-t border-slate-800/80">
          <div className="bg-slate-950/70 rounded-xl p-3.5 border border-slate-800/80">
            <div className="text-xs text-slate-400">Total Records</div>
            <div className="text-xl font-bold font-mono text-white mt-0.5">
              {pipeline.totalRows.toLocaleString()}
            </div>
          </div>
          <div className="bg-slate-950/70 rounded-xl p-3.5 border border-slate-800/80">
            <div className="text-xs text-slate-400">Input Features</div>
            <div className="text-xl font-bold font-mono text-cyan-300 mt-0.5">8 Ingredients</div>
          </div>
          <div className="bg-slate-950/70 rounded-xl p-3.5 border border-slate-800/80">
            <div className="text-xs text-slate-400">Target Variable</div>
            <div className="text-xl font-bold font-mono text-purple-300 mt-0.5">Strength (MPa)</div>
          </div>
          <div className="bg-slate-950/70 rounded-xl p-3.5 border border-slate-800/80">
            <div className="text-xs text-slate-400">Missing Values</div>
            <div className="text-xl font-bold font-mono text-emerald-400 mt-0.5">
              {pipeline.missingValues} (100% Clean)
            </div>
          </div>
        </div>
      </div>

      {/* Sub-Tab Switcher (One Primary Visual at a Time) */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-xl bg-slate-900 border border-slate-800 p-1">
          <button
            onClick={() => setActiveTab('distributions')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'distributions'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            1. Feature Distributions
          </button>
          <button
            onClick={() => setActiveTab('correlation')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'correlation'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Grid className="w-4 h-4" />
            2. Correlation Heatmap
          </button>
          <button
            onClick={() => setActiveTab('table')}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'table'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Table className="w-4 h-4" />
            3. Raw Data Table
          </button>
        </div>
      </div>

      {/* TAB 1: Feature Distributions */}
      {(activeTab === 'distributions' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-bold text-white">
                Ingredient &amp; Target Value Distribution
              </h3>
              <p className="text-xs text-slate-400">
                Select any variable below to inspect its histogram and range across all{' '}
                {pipeline.totalRows.toLocaleString()} samples
              </p>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {allColumns.map((col, idx) => (
                <button
                  key={col.key}
                  onClick={() => setSelectedColIdx(idx)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    selectedColIdx === idx
                      ? 'bg-cyan-500 text-slate-950 font-semibold'
                      : 'bg-slate-950 text-slate-300 hover:bg-slate-800 border border-slate-800'
                  }`}
                >
                  {col.shortName}
                </button>
              ))}
            </div>
          </div>

          {/* Selected Feature Mini Stats */}
          {selectedStats && (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Mean</div>
                <div className="text-sm font-mono font-bold text-white mt-0.5">
                  {selectedStats.mean.toFixed(2)} {selectedStats.unit}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Std Dev</div>
                <div className="text-sm font-mono font-bold text-cyan-300 mt-0.5">
                  {selectedStats.std.toFixed(2)} {selectedStats.unit}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Minimum</div>
                <div className="text-sm font-mono font-bold text-slate-200 mt-0.5">
                  {selectedStats.min.toFixed(2)}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Median (50%)</div>
                <div className="text-sm font-mono font-bold text-slate-200 mt-0.5">
                  {selectedStats.median.toFixed(2)}
                </div>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80">
                <div className="text-[11px] text-slate-400">Maximum</div>
                <div className="text-sm font-mono font-bold text-emerald-400 mt-0.5">
                  {selectedStats.max.toFixed(2)}
                </div>
              </div>
            </div>
          )}

          {/* Clean Large Histogram */}
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={selectedDist} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  angle={-20}
                  textAnchor="end"
                  height={50}
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
                <Bar dataKey="count" name="Sample Count" radius={[6, 6, 0, 0]}>
                  {selectedDist.map((_, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={selectedColIdx === 8 ? '#a855f7' : '#06b6d4'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* TAB 2: Correlation Heatmap */}
      {(activeTab === 'correlation' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div>
            <h3 className="text-base font-bold text-white">9 × 9 Pearson Correlation Matrix</h3>
            <p className="text-xs text-slate-400">
              Cyan cells show positive correlation; rose cells show negative correlation. Notice why
              PCA is helpful: Water and Superplasticizer are strongly correlated (r = -0.66).
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-xs font-mono">
              <thead>
                <tr>
                  <th className="p-2 text-left text-slate-400 border-b border-slate-800">
                    Variable
                  </th>
                  {allColumns.map((c) => (
                    <th
                      key={c.key}
                      className="p-2 text-center text-slate-300 border-b border-slate-800"
                    >
                      {c.shortName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pipeline.correlationMatrix.map((row, rIdx) => (
                  <tr key={allColumns[rIdx].key}>
                    <td className="p-2 font-sans font-semibold text-slate-200 border-b border-slate-800/60 whitespace-nowrap">
                      {allColumns[rIdx].shortName}
                    </td>
                    {row.map((val, cIdx) => (
                      <td
                        key={`${rIdx}-${cIdx}`}
                        style={getCorrelationStyle(val)}
                        className="p-2.5 text-center border border-slate-900/60 rounded"
                      >
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: Raw Dataset Preview Table */}
      {(activeTab === 'table' || isDetailedView) && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800/90 p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-bold text-white">Dataset Records Explorer</h3>
              <p className="text-xs text-slate-400">
                Showing rows {page * rowsPerPage + 1}–
                {Math.min((page + 1) * rowsPerPage, filteredRows.length)} of {filteredRows.length}
              </p>
            </div>
            <div className="relative w-full sm:w-64">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setPage(0);
                }}
                placeholder="Filter by row # or value..."
                className="w-full pl-9 pr-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
              />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-2 text-left">#</th>
                  {allColumns.map((col, idx) => (
                    <th
                      key={col.key}
                      className={`py-2.5 px-2 text-right ${
                        idx === 8 ? 'text-cyan-300 font-bold' : ''
                      }`}
                    >
                      {col.shortName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paginatedRows.map((row) => (
                  <tr
                    key={row.idx}
                    className="border-b border-slate-800/50 hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="py-2 px-2 text-slate-500">{row.idx}</td>
                    {row.values.map((val, cIdx) => (
                      <td
                        key={cIdx}
                        className={`py-2 px-2 text-right ${
                          cIdx === 8 ? 'text-cyan-300 font-semibold' : 'text-slate-200'
                        }`}
                      >
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between pt-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-3 py-1.5 rounded-lg bg-slate-800 disabled:opacity-40 text-xs text-slate-200 flex items-center gap-1"
            >
              <ChevronLeft className="w-3.5 h-3.5" /> Prev
            </button>
            <span className="text-xs font-mono text-slate-400">
              Page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-3 py-1.5 rounded-lg bg-slate-800 disabled:opacity-40 text-xs text-slate-200 flex items-center gap-1"
            >
              Next <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Expandable Drawer for Full Descriptive Statistics Table (df.describe()) */}
      <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowStatsDrawer(!showStatsDrawer)}
          className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors"
        >
          <span className="text-sm font-semibold text-slate-200">
            Full Summary Statistics Table (df.describe() — Mean, Std, Min, Quartiles, Max)
          </span>
          <span className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            {showStatsDrawer ? 'Hide Table' : 'Show Full Table'}
            {showStatsDrawer ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </span>
        </button>

        {showStatsDrawer && (
          <div className="px-6 pb-6 pt-2 border-t border-slate-800/80 overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 text-left font-sans">Variable</th>
                  <th className="py-2 text-right">Mean</th>
                  <th className="py-2 text-right">Std</th>
                  <th className="py-2 text-right">Min</th>
                  <th className="py-2 text-right">25%</th>
                  <th className="py-2 text-right">50%</th>
                  <th className="py-2 text-right">75%</th>
                  <th className="py-2 text-right">Max</th>
                </tr>
              </thead>
              <tbody>
                {pipeline.columnStats.map((stat) => (
                  <tr key={stat.key} className="border-b border-slate-800/50">
                    <td className="py-2 font-sans font-medium text-slate-200">
                      {stat.name} ({stat.unit})
                    </td>
                    <td className="py-2 text-right text-slate-300">{stat.mean.toFixed(2)}</td>
                    <td className="py-2 text-right text-cyan-300">{stat.std.toFixed(2)}</td>
                    <td className="py-2 text-right text-slate-400">{stat.min.toFixed(2)}</td>
                    <td className="py-2 text-right text-slate-400">{stat.q25.toFixed(2)}</td>
                    <td className="py-2 text-right text-white">{stat.median.toFixed(2)}</td>
                    <td className="py-2 text-right text-slate-400">{stat.q75.toFixed(2)}</td>
                    <td className="py-2 text-right text-emerald-400">{stat.max.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
