import {
  Atom,
  BarChart3,
  Download,
  FileText,
  Gauge,
  Printer,
  Shield,
} from "lucide-react";
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

function MetricCard({
  label,
  value,
  unit,
  color,
  hint,
}: {
  label: string;
  value: string;
  unit?: string;
  color?: string;
  hint?: string;
}) {
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

function CompactChart({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="report-chart-section compact">
      <span className="report-chart-title">{title}</span>
      <div className="report-chart-wrapper">{children}</div>
    </div>
  );
}

type ReportTabsProps = {
  playback: Playback;
  overlays?: PlaybackOverlays;
  onExportCSV: () => void;
  onExportJSON: () => void;
  onExportROS: () => void;
  onExportTXT: () => void;
  onExportPDF: () => void;
};

const TABS = [
  { key: "metrics", label: "Metrics", icon: <Gauge size={14} /> },
  { key: "charts", label: "Charts", icon: <BarChart3 size={14} /> },
  { key: "benchmarks", label: "Benchmarks", icon: <Atom size={14} /> },
  { key: "export", label: "", icon: <Download size={14} /> },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export function ReportTabs({
  playback,
  overlays,
  onExportCSV,
  onExportJSON,
  onExportROS,
  onExportTXT,
  onExportPDF,
}: ReportTabsProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("metrics");

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
  const nearMissData = useMemo(() => {
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
  }, [report.nearMissTimeline]);
  const connectivityData = useMemo(() => {
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
  }, [report.connectivityTimeline]);
  const pathEffData = useMemo(
    () =>
      report.pathEfficiencyPerDrone.map((eff, i) => ({
        drone: `#${i + 1}`,
        efficiency: Math.round(eff * 10000) / 100,
      })),
    [report.pathEfficiencyPerDrone],
  );

  const safetyScoreColor =
    report.safetyScore > 80
      ? CHART_COLORS.green
      : report.safetyScore > 50
        ? CHART_COLORS.amber
        : CHART_COLORS.red;

  return (
    <div className="report-tabs">
      <div className="report-tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`report-tab-btn${activeTab === tab.key ? " active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      <div className="report-tab-content">
        {activeTab === "metrics" && (
          <div className="report-tab-metrics">
            <MetricCard label="Avg Speed" value={report.avgSpeed.toFixed(2)} unit="m/s" color={CHART_COLORS.blue} hint="Mean speed across all drones and frames" />
            <MetricCard label="Max Speed" value={report.maxSpeed.toFixed(2)} unit="m/s" color={CHART_COLORS.purple} hint="Maximum speed observed" />
            <MetricCard label="Min Distance" value={report.minDistance.toFixed(3)} unit="m" hint="Closest approach between any two drones" />
            <MetricCard label="Collisions" value={report.totalCollisions.toLocaleString()} color={CHART_COLORS.red} hint="Total collision events (pairs within 0.3m)" />
            <MetricCard label="Near Misses" value={report.nearMisses.toLocaleString()} color={CHART_COLORS.amber} hint="Drone pairs within 0.3–0.6m" />
            <MetricCard label="Formation Error" value={report.formationError.toFixed(3)} unit="m" hint="Mean distance from swarm centroid" />
            <MetricCard label="Path Efficiency" value={`${(report.pathEfficiency * 100).toFixed(1)}%`} hint="Straight-line / actual path ratio" />
            <MetricCard label="Coverage %" value={`${report.coveragePercent}%`} color={CHART_COLORS.cyan} hint="Fraction of flight area grid visited" />
            <MetricCard label="Connectivity" value={`${report.connectivityDensity}%`} color={CHART_COLORS.green} hint="Pairs within communication range (3m)" />
            <MetricCard label="Energy Usage" value={report.energyMetric.toFixed(2)} unit="m²/s²" hint="Mean squared speed (kinetic energy proxy)" />
            <MetricCard label="Safety Score" value={`${report.safetyScore}%`} color={safetyScoreColor} hint="Frames without collisions" />
            <MetricCard label="Duration" value={formatDuration(report.flightDuration)} hint="Total simulated time" />
            <MetricCard label="Total Distance" value={report.totalDistance.toFixed(1)} unit="m" hint="Cumulative distance all drones" />
          </div>
        )}

        {activeTab === "charts" && (
          <div className="report-tab-charts">
            <div className="report-section-heading">Performance</div>
            <CompactChart title="Mean Speed over Time">
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={speedData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v).toFixed(2)} m/s`, "Speed"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Area type="monotone" dataKey="value" stroke={CHART_COLORS.blue} fill={CHART_COLORS.blue} fillOpacity={0.12} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </CompactChart>

            <CompactChart title="Formation Error over Time">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={formationData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v).toFixed(3)} m`, "Error"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Line type="monotone" dataKey="value" stroke={CHART_COLORS.purple} strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CompactChart>

            <CompactChart title="Energy Usage">
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={speedData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${(Number(v) * Number(v)).toFixed(2)} m²/s²`, "Energy"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Area type="monotone" dataKey="value" stroke={CHART_COLORS.cyan} fill={CHART_COLORS.cyan} fillOpacity={0.1} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </CompactChart>

            <CompactChart title="Path Efficiency per Drone">
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={pathEffData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="drone" tick={{ fontSize: 8, fill: "#94a3b8" }} tickLine={false} axisLine={false} interval={Math.max(0, Math.floor(pathEffData.length / 12))} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={36} domain={[0, 100]} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v).toFixed(1)}%`, "Efficiency"]} />
                  <Bar dataKey="efficiency" fill={CHART_COLORS.blue} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CompactChart>

            <div className="report-section-heading">Safety &amp; Communication</div>

            <CompactChart title="Collisions per Frame">
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={collisionData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={30} allowDecimals={false} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v)}`, "Collisions"]} labelFormatter={(l) => `t = ${l}s`} />
                  <Area type="stepAfter" dataKey="value" stroke={CHART_COLORS.red} fill={CHART_COLORS.red} fillOpacity={0.15} strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </CompactChart>

            <CompactChart title="Near Misses (0.3–0.6m)">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={nearMissData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="index" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={36} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v)}`, "Near Misses"]} />
                  <Line type="monotone" dataKey="value" stroke={CHART_COLORS.amber} strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CompactChart>

            <CompactChart title="Connectivity Density">
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={connectivityData} margin={{ top: 4, right: 4, left: -8, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="index" tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={36} unit="%" />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e2e8f0" }} formatter={(v) => [`${Number(v)}%`, "Connected"]} />
                  <Line type="monotone" dataKey="value" stroke={CHART_COLORS.green} strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </CompactChart>

            <CompactChart title="Coverage &amp; Safety Overview">
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
            </CompactChart>
          </div>
        )}

        {activeTab === "benchmarks" && (
          <div className="report-tab-benchmarks">
            <BenchmarkCard
              gpuMetrics={playback.gpuMetrics}
              deviceInfo={playback.deviceInfo}
              gpuPlatform={playback.gpuPlatform}
            />
          </div>
        )}

        {activeTab === "export" && (
          <div className="report-tab-export">
            <div className="report-section-heading">
              <Shield size={14} />
              Report Exports
            </div>
            <div className="report-tab-export-grid">
              <button className="report-export-btn" onClick={onExportPDF}>
                <Printer size={18} />
                <span>PDF Report</span>
                <small>Printable summary with charts</small>
              </button>
              <button className="report-export-btn" onClick={onExportTXT}>
                <FileText size={18} />
                <span>TXT Report</span>
                <small>Plain-text summary</small>
              </button>
            </div>

            <div className="report-section-heading">
              <Download size={14} />
              Waypoint Exports
            </div>
            <div className="report-tab-export-grid">
              <button className="report-export-btn" onClick={onExportCSV}>
                <Download size={18} />
                <span>CSV Waypoints</span>
                <small>Tabular drone positions</small>
              </button>
              <button className="report-export-btn" onClick={onExportJSON}>
                <Download size={18} />
                <span>JSON Waypoints</span>
                <small>Structured drone data</small>
              </button>
              <button className="report-export-btn" onClick={onExportROS}>
                <Download size={18} />
                <span>ROS Waypoints</span>
                <small>Robot Operating System format</small>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
