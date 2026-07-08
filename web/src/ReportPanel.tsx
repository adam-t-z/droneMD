import { ChevronDown, ChevronRight, FileText, Printer } from "lucide-react";
import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BenchmarkCard } from "./BenchmarkCard";
import { computeReport, formatDuration } from "./export";
import type { Playback, PlaybackOverlays } from "./types";

const CHART_MAX_POINTS = 300;

function downsampleWithTime(
  data: number[],
  timestamps: number[],
  maxPoints: number,
): { time: number; value: number }[] {
  const n = data.length;
  if (n <= maxPoints) {
    return data.map((v, i) => ({ time: Math.round(timestamps[i] * 10) / 10, value: v }));
  }
  const step = n / maxPoints;
  const result: { time: number; value: number }[] = [];
  for (let i = 0; i < maxPoints; i++) {
    const idx = Math.floor(i * step);
    result.push({ time: Math.round(timestamps[idx] * 10) / 10, value: data[idx] });
  }
  return result;
}

const CHART_COLORS = {
  blue: "#2563eb",
  red: "#dc2626",
  amber: "#d97706",
  green: "#16a34a",
  purple: "#7c3aed",
  cyan: "#0891b2",
  slate: "#64748b",
};

type MetricCardProps = {
  label: string;
  value: string;
  unit?: string;
  color?: string;
  hint?: string;
};

function MetricCard({ label, value, unit, color, hint }: MetricCardProps) {
  return (
    <div className="report-card" title={hint}>
      <span className="report-card-label">{label}</span>
      <span className="report-card-value" style={color ? { color } : undefined}>
        {value}
        {unit && <span className="report-card-unit"> {unit}</span>}
      </span>
    </div>
  );
}

type ChartSectionProps = {
  title: string;
  children: React.ReactNode;
};

function ChartSection({ title, children }: ChartSectionProps) {
  return (
    <div className="report-chart-section">
      <span className="report-chart-title">{title}</span>
      <div className="report-chart-wrapper">{children}</div>
    </div>
  );
}

type ReportPanelProps = {
  playback: Playback;
  overlays?: PlaybackOverlays;
  onExportCSV: () => void;
  onExportJSON: () => void;
  onExportROS: () => void;
  onExportTXT: () => void;
  onExportPDF: () => void;
};

export function ReportPanel({
  playback,
  overlays,
  onExportCSV,
  onExportJSON,
  onExportROS,
  onExportTXT,
  onExportPDF,
}: ReportPanelProps) {
  const [isOpen, setIsOpen] = useState(true);

  const report = useMemo(() => computeReport(playback, overlays), [playback, overlays]);

  const speedData = useMemo(
    () => downsampleWithTime(report.speedTimeline, playback.timestamps, CHART_MAX_POINTS),
    [report.speedTimeline, playback.timestamps],
  );
  const formationData = useMemo(
    () => downsampleWithTime(report.formationErrorTimeline, playback.timestamps, CHART_MAX_POINTS),
    [report.formationErrorTimeline, playback.timestamps],
  );
  const collisionData = useMemo(
    () => downsampleWithTime(report.collisionTimeline, playback.timestamps, CHART_MAX_POINTS),
    [report.collisionTimeline, playback.timestamps],
  );
  const nearMissData = useMemo(
    () => {
      const sampled = report.nearMissTimeline;
      const n = sampled.length;
      if (n <= CHART_MAX_POINTS) {
        return sampled.map((v, i) => ({ index: i, value: v }));
      }
      const step = n / CHART_MAX_POINTS;
      const result: { index: number; value: number }[] = [];
      for (let i = 0; i < CHART_MAX_POINTS; i++) {
        const idx = Math.floor(i * step);
        result.push({ index: idx, value: sampled[idx] });
      }
      return result;
    },
    [report.nearMissTimeline],
  );
  const connectivityData = useMemo(
    () => {
      const sampled = report.connectivityTimeline;
      const n = sampled.length;
      if (n <= CHART_MAX_POINTS) {
        return sampled.map((v, i) => ({ index: i, value: v }));
      }
      const step = n / CHART_MAX_POINTS;
      const result: { index: number; value: number }[] = [];
      for (let i = 0; i < CHART_MAX_POINTS; i++) {
        const idx = Math.floor(i * step);
        result.push({ index: idx, value: sampled[idx] });
      }
      return result;
    },
    [report.connectivityTimeline],
  );
  const pathEffData = useMemo(
    () =>
      report.pathEfficiencyPerDrone.map((eff, i) => ({
        drone: `#${i + 1}`,
        efficiency: Math.round(eff * 10000) / 100,
      })),
    [report.pathEfficiencyPerDrone],
  );

  const safetyScoreColor =
    report.safetyScore > 80 ? CHART_COLORS.green : report.safetyScore > 50 ? CHART_COLORS.amber : CHART_COLORS.red;

  return (
    <div className="report-panel">
      <button className="report-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
        <span className="eyebrow">Simulation Report</span>
      </button>

      {isOpen && (
        <div className="report-body">
          {/* Metric cards */}
          <div className="report-grid">
            <MetricCard label="Avg Speed" value={report.avgSpeed.toFixed(2)} unit="m/s" color={CHART_COLORS.blue} hint="Mean speed across all drones and frames" />
            <MetricCard label="Max Speed" value={report.maxSpeed.toFixed(2)} unit="m/s" color={CHART_COLORS.purple} hint="Maximum speed observed" />
            <MetricCard label="Min Distance" value={report.minDistance.toFixed(3)} unit="m" hint="Closest approach between any two drones" />
            <MetricCard label="Collisions" value={report.totalCollisions.toLocaleString()} color={CHART_COLORS.red} hint="Total collision events (pairs within 0.3m)" />
            <MetricCard label="Near Misses" value={report.nearMisses.toLocaleString()} color={CHART_COLORS.amber} hint="Drone pairs within 0.3&ndash;0.6m" />
            <MetricCard label="Formation Error" value={report.formationError.toFixed(3)} unit="m" hint="Mean distance from swarm centroid" />
            <MetricCard
              label="Path Efficiency"
              value={`${(report.pathEfficiency * 100).toFixed(1)}%`}
              hint="Straight-line / actual path ratio"
            />
            <MetricCard label="Coverage %" value={`${report.coveragePercent}%`} color={CHART_COLORS.cyan} hint="Fraction of flight area grid visited" />
            <MetricCard
              label="Connectivity"
              value={`${report.connectivityDensity}%`}
              color={CHART_COLORS.green}
              hint="Pairs within communication range (3m)"
            />
            <MetricCard
              label="Energy Usage"
              value={report.energyMetric.toFixed(2)}
              unit="m²/s²"
              hint="Mean squared speed (kinetic energy proxy)"
            />
            <MetricCard
              label="Safety Score"
              value={`${report.safetyScore}%`}
              color={safetyScoreColor}
              hint="Frames without collisions"
            />
            <MetricCard label="Duration" value={formatDuration(report.flightDuration)} hint="Total simulated time" />
            <MetricCard label="Total Distance" value={report.totalDistance.toFixed(1)} unit="m" hint="Cumulative distance all drones" />
          </div>

          {/* Part 1 — Performance */}
          <h3 className="report-section-heading">Performance</h3>
          <div className="report-charts-grid">
            <ChartSection title="Mean Speed over Time">
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={speedData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} label={{ value: "s", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v).toFixed(2)} m/s`, "Mean Speed"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Area type="monotone" dataKey="value" stroke={CHART_COLORS.blue} fill={CHART_COLORS.blue} fillOpacity={0.12} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartSection>

            <ChartSection title="Formation Error over Time">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={formationData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} label={{ value: "s", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={44} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v).toFixed(3)} m`, "Formation Error"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Line type="monotone" dataKey="value" stroke={CHART_COLORS.purple} strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartSection>

            <ChartSection title="Energy Usage">
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={speedData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} label={{ value: "s", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={44} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${(Number(v) * Number(v)).toFixed(2)} m²/s²`, "Energy"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Area type="monotone" dataKey="value" stroke={CHART_COLORS.cyan} fill={CHART_COLORS.cyan} fillOpacity={0.1} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartSection>

            <ChartSection title="Path Efficiency per Drone">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={pathEffData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="drone" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} interval={Math.max(0, Math.floor(pathEffData.length / 15))} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={40} domain={[0, 100]} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v).toFixed(1)}%`, "Efficiency"]} />
                  <Bar dataKey="efficiency" fill={CHART_COLORS.blue} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartSection>
          </div>

          {/* Part 2 — Safety & Communication */}
          <h3 className="report-section-heading">Safety &amp; Communication</h3>
          <div className="report-charts-grid">
            <ChartSection title="Collisions per Frame">
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={collisionData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} label={{ value: "s", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={36} allowDecimals={false} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v)}`, "Collisions"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Area type="stepAfter" dataKey="value" stroke={CHART_COLORS.red} fill={CHART_COLORS.red} fillOpacity={0.15} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </ChartSection>

            <ChartSection title="Near Misses per Frame (0.3&ndash;0.6m)">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={nearMissData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="index" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} label={{ value: "sample", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v)}`, "Near Misses"]} />
                  <Line type="monotone" dataKey="value" stroke={CHART_COLORS.amber} strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartSection>

            <ChartSection title="Connectivity Density">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={connectivityData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="index" tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} label={{ value: "sample", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={40} unit="%" />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v)}%`, "Connected"]} />
                  <Line type="monotone" dataKey="value" stroke={CHART_COLORS.green} strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartSection>

            <ChartSection title="Coverage &amp; Safety Overview">
              <div className="report-coverage-summary">
                <div className="report-coverage-stat">
                  <span className="report-coverage-value">{report.coveragePercent}%</span>
                  <span className="report-coverage-label">Area Coverage</span>
                </div>
                <div className="report-coverage-stat">
                  <span className="report-coverage-value" style={{ color: safetyScoreColor }}>{report.safetyScore}%</span>
                  <span className="report-coverage-label">Safety Score</span>
                </div>
                <div className="report-coverage-stat">
                  <span className="report-coverage-value" style={{ color: CHART_COLORS.amber }}>{report.nearMisses.toLocaleString()}</span>
                  <span className="report-coverage-label">Near Misses</span>
                </div>
              </div>
            </ChartSection>
          </div>

          {/* Part 3 — GPU Benchmarks */}
          <h3 className="report-section-heading">GPU Benchmarks</h3>
          <BenchmarkCard
            gpuMetrics={playback.gpuMetrics}
            deviceInfo={playback.deviceInfo}
            gpuPlatform={playback.gpuPlatform}
          />

          {/* Export actions */}
          <div className="report-actions">
            <button className="secondary-action compact" onClick={onExportPDF}>
              <Printer size={14} />
              PDF
            </button>
            <button className="secondary-action compact" onClick={onExportTXT}>
              <FileText size={14} />
              TXT
            </button>
            <span className="report-actions-separator" />
            <button className="secondary-action compact" onClick={onExportCSV}>CSV</button>
            <button className="secondary-action compact" onClick={onExportJSON}>JSON</button>
            <button className="secondary-action compact" onClick={onExportROS}>ROS</button>
          </div>
        </div>
      )}
    </div>
  );
}
