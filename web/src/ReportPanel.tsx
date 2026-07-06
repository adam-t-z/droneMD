import { ChevronDown, ChevronRight, FileText, Printer } from "lucide-react";
import { useMemo, useState } from "react";
import { computeReport, formatDuration } from "./export";
import type { Playback, PlaybackOverlays } from "./types";

type MetricCardProps = {
  label: string;
  value: string;
  unit?: string;
  color?: string;
};

function MetricCard({ label, value, unit, color }: MetricCardProps) {
  return (
    <div className="report-card">
      <span className="report-card-label">{label}</span>
      <span className="report-card-value" style={color ? { color } : undefined}>
        {value}
        {unit && <span className="report-card-unit"> {unit}</span>}
      </span>
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

export function ReportPanel({ playback, overlays, onExportCSV, onExportJSON, onExportROS, onExportTXT, onExportPDF }: ReportPanelProps) {
  const [isOpen, setIsOpen] = useState(true);
  const report = useMemo(() => computeReport(playback, overlays), [playback, overlays]);

  const maxCollisions = Math.max(...report.collisionTimeline, 1);

  return (
    <div className="report-panel">
      <button className="report-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
        <span className="eyebrow">Simulation Report</span>
      </button>

      {isOpen && (
        <div className="report-body">
          <div className="report-grid">
            <MetricCard label="Total Collisions" value={report.totalCollisions.toLocaleString()} color="#dc2626" />
            <MetricCard label="Avg Speed" value={report.avgSpeed.toFixed(2)} unit="m/s" />
            <MetricCard label="Max Speed" value={report.maxSpeed.toFixed(2)} unit="m/s" color="#2563eb" />
            <MetricCard label="Min Speed" value={report.minSpeed.toFixed(2)} unit="m/s" />
            <MetricCard label="Safety Score" value={`${report.safetyScore}%`} color={report.safetyScore > 80 ? "#16a34a" : report.safetyScore > 50 ? "#d97706" : "#dc2626"} />
            <MetricCard label="Flight Duration" value={formatDuration(report.flightDuration)} />
            <MetricCard label="Total Distance" value={report.totalDistance.toFixed(1)} unit="m" />
            <MetricCard label="Energy Metric" value={report.energyMetric.toFixed(2)} unit="m²/s²" />
          </div>

          {report.collisionTimeline.length > 0 && (
            <div className="report-timeline">
              <span className="report-card-label">Collision Timeline</span>
              <div className="report-timeline-bars">
                {report.collisionTimeline.map((count, i) => {
                  const barPct = (count / maxCollisions) * 100;
                  const freq = report.collisionTimeline.length;
                  const skip = Math.max(1, Math.floor(freq / 50));
                  if (i % skip !== 0) return null;
                  return (
                    <div
                      key={i}
                      className="report-timeline-bar"
                      style={{ height: `${Math.max(2, barPct)}%` }}
                      title={`t=${(i / (freq / playback.timestamps[playback.timestamps.length - 1])).toFixed(1)}s: ${count} collisions`}
                    />
                  );
                })}
              </div>
            </div>
          )}

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
