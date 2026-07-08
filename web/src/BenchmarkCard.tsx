import { Cpu, Gauge, HardDrive, MemoryStick, Microchip, Timer, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchGpuBenchmark } from "./api";
import type { BenchmarkHistory, DeviceInfo, GpuMetrics } from "./types";

const PLATFORM_COLORS: Record<string, { bg: string; text: string; badge: string }> = {
  rocm: { bg: "#fef2f2", text: "#dc2626", badge: "#dc2626" },
  cuda: { bg: "#ecfdf5", text: "#16a34a", badge: "#16a34a" },
  cpu: { bg: "#f1f5f9", text: "#64748b", badge: "#94a3b8" },
};

const PLATFORM_LABELS: Record<string, string> = {
  rocm: "AMD ROCm",
  cuda: "NVIDIA CUDA",
  cpu: "CPU",
};

function StatCard({
  icon,
  label,
  value,
  unit,
  hint,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit?: string;
  hint?: string;
}) {
  return (
    <div className="report-card" title={hint}>
      <span className="report-card-label">{label}</span>
      <span className="report-card-value">
        {icon}
        {value}
        {unit && <span className="report-card-unit"> {unit}</span>}
      </span>
    </div>
  );
}

function HistoryTable({ measurements }: { measurements: GpuMetrics[] }) {
  if (measurements.length === 0) return null;

  return (
    <div className="bench-history-section">
      <div className="bench-history-header">
        <span className="report-card-label">Recent Benchmark Runs</span>
        <span className="bench-history-count">{measurements.length} runs</span>
      </div>
      <div className="bench-history-table-wrap">
        <table className="bench-history-table">
          <thead>
            <tr>
              <th>Drones</th>
              <th>Duration</th>
              <th>Wall Time</th>
              <th>Steps/s</th>
              <th>Memory</th>
              <th>Platform</th>
            </tr>
          </thead>
          <tbody>
            {measurements.slice(-10).reverse().map((m, i) => (
              <tr key={i}>
                <td>{m.num_drones}</td>
                <td>{m.duration_seconds}s</td>
                <td>{m.sim_time_seconds.toFixed(1)}s</td>
                <td className="bench-mono">{m.timesteps_per_second.toLocaleString()}</td>
                <td className="bench-mono">
                  {m.device_memory_mb != null ? `${m.device_memory_mb} MB` : "—"}
                </td>
                <td>
                  <span className="bench-platform-badge">{PLATFORM_LABELS[m.platform] ?? m.platform}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PlatformBanner({ platform, deviceInfo }: { platform: string; deviceInfo?: DeviceInfo }) {
  const colors = PLATFORM_COLORS[platform] ?? PLATFORM_COLORS.cpu;
  const label = PLATFORM_LABELS[platform] ?? platform.toUpperCase();
  const deviceName = deviceInfo?.device_name ?? platform.toUpperCase();

  return (
    <div
      className="bench-banner"
      style={{ borderColor: colors.badge, background: colors.bg }}
    >
      <div className="bench-banner-left">
        <span
          className="bench-platform-pill"
          style={{ background: colors.badge, color: "#fff" }}
        >
          {label}
        </span>
        <span className="bench-device-name">{deviceName}</span>
      </div>
      <div className="bench-banner-right">
        {platform === "rocm" && (
          <span className="bench-tag" style={{ background: "#fef3c7", color: "#92400e" }}>
            MI300X 192 GB HBM3
          </span>
        )}
      </div>
    </div>
  );
}

interface BenchmarkCardProps {
  gpuMetrics?: GpuMetrics;
  deviceInfo?: DeviceInfo;
  gpuPlatform?: string;
}

export function BenchmarkCard({ gpuMetrics, deviceInfo, gpuPlatform }: BenchmarkCardProps) {
  const [history, setHistory] = useState<BenchmarkHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    setHistoryLoading(true);
    fetchGpuBenchmark()
      .then((data) => setHistory(data))
      .catch(() => setHistory(null))
      .finally(() => setHistoryLoading(false));
  }, []);

  const platform = gpuPlatform ?? gpuMetrics?.platform ?? "cpu";
  const metrics = gpuMetrics;

  if (!metrics) {
    return (
      <div className="bench-empty">
        <Microchip size={24} />
        <p>Run a simulation to see GPU benchmark data.</p>
      </div>
    );
  }

  const platformIcon =
    platform === "rocm" ? <Zap size={16} /> : platform === "cuda" ? <Gauge size={16} /> : <Cpu size={16} />;

  return (
    <div className="bench-section">
      <PlatformBanner platform={platform} deviceInfo={deviceInfo} />

      <div className="report-grid">
        <StatCard
          icon={platformIcon}
          label="Compute"
          value={PLATFORM_LABELS[platform] ?? platform.toUpperCase()}
          hint={`Platform: ${metrics.platform}`}
        />
        <StatCard
          icon={<Timer size={16} />}
          label="Wall Time"
          value={metrics.sim_time_seconds.toFixed(1)}
          unit="s"
          hint="Total simulation wall-clock time"
        />
        <StatCard
          icon={<Gauge size={16} />}
          label="Throughput"
          value={metrics.timesteps_per_second.toLocaleString()}
          unit="steps/s"
          hint="Physics steps per wall-second"
        />
        <StatCard
          icon={<HardDrive size={16} />}
          label="Process Memory"
          value={metrics.device_memory_mb != null ? `${metrics.device_memory_mb}` : "—"}
          unit={metrics.device_memory_mb != null ? "MB" : undefined}
          hint="Resident memory (RSS)"
        />
        <StatCard
          icon={<Cpu size={16} />}
          label="Drone Count"
          value={metrics.num_drones.toLocaleString()}
          hint="Number of simulated drones"
        />
        <StatCard
          icon={<Timer size={16} />}
          label="Duration"
          value={metrics.duration_seconds.toFixed(0)}
          unit="s"
          hint="Simulated time"
        />
        <StatCard
          icon={<MemoryStick size={16} />}
          label="Physics Freq"
          value={metrics.physics_freq_hz.toLocaleString()}
          unit="Hz"
          hint="Physics simulation frequency"
        />
        <StatCard
          icon={<MemoryStick size={16} />}
          label="Control Freq"
          value={metrics.control_freq_hz.toLocaleString()}
          unit="Hz"
          hint="State control update frequency"
        />
      </div>

      {historyLoading && (
        <div className="bench-loading">Loading benchmark history...</div>
      )}

      {history && history.measurements.length > 0 && (
        <HistoryTable measurements={history.measurements} />
      )}

      {history && !historyLoading && history.measurements.length === 0 && (
        <div className="bench-why-section">
          <span className="report-card-label">Why AMD MI300X</span>
          <p>
            AMD MI300X's 192 GB HBM3 memory enables simultaneous simulation of 200+
            physically-accurate quadrotor drones at 500 Hz physics frequency — workloads that
            exceed typical GPU memory limits. JAX's XLA compiler auto-vectorizes swarm dynamics
            across thousands of ROCm compute units, achieving near-linear scaling.
          </p>
        </div>
      )}
    </div>
  );
}
