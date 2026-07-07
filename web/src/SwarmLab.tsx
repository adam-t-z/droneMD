import { Atom, ChevronDown, ChevronRight, Download, FlaskConical, Loader2, Maximize2, Minimize2, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { simulateSwarmStream } from "./api";
import { downloadCSV, downloadJSONWaypoints, downloadROS, downloadReportPdf, downloadReportTxt } from "./export";
import { Player } from "./Player";
import { ReportPanel } from "./ReportPanel";
import type { Playback, PlaybackOverlays, SimPhase, SwarmConfig } from "./types";

const DEFAULT_CONFIG: SwarmConfig = {
  n_drones: 15,
  duration: 20,
  separation_weight: 1.5,
  alignment_weight: 1.0,
  cohesion_weight: 1.0,
  perception_radius: 3.0,
  max_speed: 2.0,
  max_force: 0.5,
  boundary_mode: "wrap",
  bounds: [-2.0, 2.0, -2.0, 2.0],
  obstacles: [],
  device: "cpu",
  physics: "first_principles",
  integrator: "euler",
  freq: 500,
  state_freq: 100,
  height: 1.0,
  motion_primitive: "none",
  primitive_params: {},
};

const PHASES: { key: string; label: string }[] = [
  { key: "Initializing simulation engine", label: "Initializing" },
  { key: "Running flocking simulation", label: "Simulating" },
  { key: "Computing collision and speed data", label: "Computing" },
  { key: "Finalizing playback data", label: "Finalizing" },
];

type SliderDef = {
  key: keyof SwarmConfig;
  label: string;
  min: number;
  max: number;
  step: number;
};

const DRONE_SLIDERS: SliderDef[] = [
  { key: "n_drones", label: "Drone Count", min: 10, max: 200, step: 1 },
  { key: "duration", label: "Duration (s)", min: 5, max: 120, step: 1 },
];

const BEHAVIOR_SLIDERS: SliderDef[] = [
  { key: "separation_weight", label: "Separation", min: 0, max: 5, step: 0.1 },
  { key: "alignment_weight", label: "Alignment", min: 0, max: 5, step: 0.1 },
  { key: "cohesion_weight", label: "Cohesion", min: 0, max: 5, step: 0.1 },
  { key: "perception_radius", label: "Perception Radius", min: 0.5, max: 10, step: 0.1 },
  { key: "max_speed", label: "Max Speed", min: 0.5, max: 5, step: 0.1 },
  { key: "max_force", label: "Max Force", min: 0.1, max: 3, step: 0.1 },
];

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="swarm-slider">
      <span>{label}</span>
      <div className="swarm-slider-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <span className="swarm-slider-value">{value}</span>
      </div>
    </label>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function ProgressIndicator({ phase, percent }: { phase: string; percent: number }) {
  const activeIdx = PHASES.findIndex((p) => p.key === phase);
  const activePhase = PHASES[activeIdx];

  return (
    <div className="swarm-progress">
      <div className="swarm-progress-header">
        <Loader2 size={18} className="spin" />
        <span className="swarm-progress-phase">{activePhase?.label ?? phase}</span>
        <span className="swarm-progress-pct">{percent}%</span>
      </div>
      <div className="swarm-progress-track">
        <div className="swarm-progress-fill" style={{ width: `${percent}%` }} />
      </div>
      <div className="swarm-progress-steps">
        {PHASES.map((p, i) => {
          const done = i < activeIdx;
          const current = i === activeIdx;
          return (
            <div
              key={p.key}
              className={`swarm-progress-step${done ? " done" : ""}${current ? " current" : ""}`}
            >
              <span className="swarm-progress-dot" />
              <span className="swarm-progress-step-label">{p.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SwarmLab() {
  const [config, setConfig] = useState<SwarmConfig>(DEFAULT_CONFIG);
  const [simPhase, setSimPhase] = useState<SimPhase | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playback, setPlayback] = useState<Playback | null>(null);
  const [overlays, setOverlays] = useState<PlaybackOverlays | null>(null);
  const [showBehavior, setShowBehavior] = useState(false);
  const [showEnvironment, setShowEnvironment] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const previewRef = useRef<HTMLDivElement>(null);

  const updateConfig = useCallback(<K extends keyof SwarmConfig>(
    key: K,
    value: SwarmConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }, []);

  const updatePrimitiveParam = useCallback((key: string, value: unknown) => {
    setConfig((prev) => ({
      ...prev,
      primitive_params: { ...prev.primitive_params, [key]: value },
    }));
  }, []);

  const getPrimitiveParam = useCallback(
    (key: string, defaultVal: number): number => {
      const v = config.primitive_params[key];
      return typeof v === "number" ? v : defaultVal;
    },
    [config.primitive_params],
  );

  const startSimulation = useCallback(async () => {
    setLoading(true);
    setSimPhase({ phase: "Initializing simulation engine", percent: 0 });
    setError(null);
    try {
      const result = await simulateSwarmStream(config, (phase) => {
        setSimPhase(phase);
      });
      const { overlays: ov, ...rest } = result;
      setPlayback(rest);
      setOverlays(ov ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      setLoading(false);
      setSimPhase(null);
    }
  }, [config]);

  const reset = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
    setError(null);
    setPlayback(null);
    setOverlays(null);
    setSimPhase(null);
  }, []);

  const togglePreviewFullscreen = useCallback(async () => {
    if (!previewRef.current) return;

    if (!document.fullscreenElement) {
      await previewRef.current.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const statusLabel = loading ? "Simulating" : playback ? "Complete" : "Ready";
  const statusClass = loading
    ? "status-pill playing"
    : playback
    ? "status-pill playing"
    : "status-pill";

  return (
    <main className="app-shell swarm-lab">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">DroneMD</p>
            <h1>DroneMD</h1>
            <p className="swarm-header-description">
              This is a drone swarm simulation system running on AMD GPU hardware, with live controls for swarm behavior, with actual physics engine.
            </p>
            <div className="swarm-status-row">
              <span className={statusClass}>{statusLabel}</span>
              <span className="meta-chip">{config.device.toUpperCase()}</span>
              <span className="meta-chip">{config.n_drones} drones</span>
              <span className="meta-chip">{config.duration}s</span>
            </div>
          </div>
          <div className="topbar-actions">
            <button className="secondary-action compact" onClick={reset}>
              <RotateCcw size={18} />
              Reset
            </button>
          </div>
        </header>

        <div className="swarm-layout">
          <div className="swarm-controls">
            <div className="swarm-section">
              <div className="section-title">
                <Atom size={18} />
                <h2>Drones</h2>
              </div>
              {DRONE_SLIDERS.map((s) => (
                <Slider
                  key={s.key}
                  label={s.label}
                  value={config[s.key] as number}
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  onChange={(v) => updateConfig(s.key as keyof SwarmConfig, v)}
                />
              ))}
            </div>

            

            <div className="swarm-section">
              <div className="section-title">
                <h2>Motion Primitive</h2>
              </div>
              <label className="swarm-slider">
                <span>Shape</span>
                <select
                  value={config.motion_primitive}
                  onChange={(e) =>
                    updateConfig("motion_primitive", e.target.value as SwarmConfig["motion_primitive"])
                  }
                >
                  <option value="none">None (Flocking)</option>
                  <option value="circle">Circle</option>
                  <option value="star">Star</option>
                  <option value="cone">Cone</option>
                </select>
              </label>
              {config.motion_primitive === "circle" && (
                <>
                  <Slider
                    label="Radius"
                    value={getPrimitiveParam("radius", 1.5)}
                    min={0.3}
                    max={2.0}
                    step={0.1}
                    onChange={(v) => updatePrimitiveParam("radius", v)}
                  />
                  <Slider
                    label="Rotation (rad/s)"
                    value={getPrimitiveParam("rotation", 0.3)}
                    min={0}
                    max={1.5}
                    step={0.1}
                    onChange={(v) => updatePrimitiveParam("rotation", v)}
                  />
                  <p className="spawn-csv-hint">
                    Drones form a circle and hold the shape continuously (rotate at 0 to hold still).
                  </p>
                </>
              )}
              {config.motion_primitive === "star" && (
                <>
                  <Slider
                    label="Inner Radius"
                    value={getPrimitiveParam("radius", 1.2)}
                    min={0.3}
                    max={2.0}
                    step={0.1}
                    onChange={(v) => updatePrimitiveParam("radius", v)}
                  />
                  <Slider
                    label="Spoke Gap"
                    value={getPrimitiveParam("delta_radius", 0.4)}
                    min={0.1}
                    max={1.0}
                    step={0.05}
                    onChange={(v) => updatePrimitiveParam("delta_radius", v)}
                  />
                  <Slider
                    label="Rotation (rad/s)"
                    value={getPrimitiveParam("rotation", 0.2)}
                    min={0}
                    max={1.5}
                    step={0.1}
                    onChange={(v) => updatePrimitiveParam("rotation", v)}
                  />
                  <p className="spawn-csv-hint">
                    Two interleaved rings form a star with n/2 spokes; shape is held continuously.
                  </p>
                </>
              )}
              {config.motion_primitive === "cone" && (
                <>
                  <Slider
                    label="Layer Height"
                    value={getPrimitiveParam("delta_height", 0.3)}
                    min={0.1}
                    max={0.8}
                    step={0.05}
                    onChange={(v) => updatePrimitiveParam("delta_height", v)}
                  />
                  <Slider
                    label="Spacing"
                    value={getPrimitiveParam("spacing", 0.5)}
                    min={0.3}
                    max={1.2}
                    step={0.05}
                    onChange={(v) => updatePrimitiveParam("spacing", v)}
                  />
                  <Slider
                    label="Rotation (rad/s)"
                    value={getPrimitiveParam("rotation", 0.3)}
                    min={0}
                    max={1.5}
                    step={0.1}
                    onChange={(v) => updatePrimitiveParam("rotation", v)}
                  />
                  <label className="swarm-slider">
                    <span>Inverted</span>
                    <input
                      type="checkbox"
                      checked={Boolean(config.primitive_params.inverted)}
                      onChange={(e) => updatePrimitiveParam("inverted", e.target.checked)}
                    />
                  </label>
                  <p className="spawn-csv-hint">
                    Layered rings stack into a cone (apex up, or down if inverted); shape is held.
                  </p>
                </>
              )}
            </div>

            <div className="swarm-section">
              <button
                className="section-title swarm-advanced-toggle"
                onClick={() => setShowBehavior(!showBehavior)}
              >
                {showBehavior ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                <FlaskConical size={18} />
                <h2>Behavior</h2>
              </button>
              {showBehavior && (
                <div className="swarm-advanced-body">
                  {BEHAVIOR_SLIDERS.map((s) => (
                    <Slider
                      key={s.key}
                      label={s.label}
                      value={config[s.key] as number}
                      min={s.min}
                      max={s.max}
                      step={s.step}
                      onChange={(v) => updateConfig(s.key as keyof SwarmConfig, v)}
                    />
                  ))}
                </div>
              )}
            </div>

            <div className="swarm-section">
              <button
                className="section-title swarm-advanced-toggle"
                onClick={() => setShowEnvironment(!showEnvironment)}
              >
                {showEnvironment ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                <h2>Environment</h2>
              </button>
              {showEnvironment && (
                <div className="swarm-advanced-body">
                  <label className="swarm-slider">
                    <span>Boundary Mode</span>
                    <select
                      value={config.boundary_mode}
                      onChange={(e) =>
                        updateConfig("boundary_mode", e.target.value as SwarmConfig["boundary_mode"])
                      }
                    >
                      <option value="wrap">Wrap</option>
                      <option value="bounce">Bounce</option>
                      <option value="hard">Hard</option>
                    </select>
                  </label>
                  <label className="swarm-slider">
                    <span>Device</span>
                    <select
                      value={config.device}
                      onChange={(e) =>
                        updateConfig("device", e.target.value as SwarmConfig["device"])
                      }
                    >
                      <option value="cpu">CPU</option>
                      <option value="gpu">GPU</option>
                    </select>
                  </label>
                </div>
              )}
            </div>

            <div className="swarm-section">
              <button
                className="section-title swarm-advanced-toggle"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                {showAdvanced ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                <h2>Advanced</h2>
              </button>
              {showAdvanced && (
                <div className="swarm-advanced-body">
                  <label className="swarm-slider">
                    <span>Physics Model</span>
                    <select
                      value={config.physics}
                      onChange={(e) =>
                        updateConfig("physics", e.target.value as SwarmConfig["physics"])
                      }
                    >
                      <option value="first_principles">First Principles</option>
                      <option value="so_rpy">SO(3) RPY</option>
                      <option value="so_rpy_rotor">SO(3) RPY Rotor</option>
                      <option value="so_rpy_rotor_drag">SO(3) RPY Rotor Drag</option>
                    </select>
                  </label>
                  <label className="swarm-slider">
                    <span>Integrator</span>
                    <select
                      value={config.integrator}
                      onChange={(e) =>
                        updateConfig("integrator", e.target.value as SwarmConfig["integrator"])
                      }
                    >
                      <option value="euler">Euler</option>
                      <option value="rk4">RK4</option>
                      <option value="symplectic_euler">Symplectic Euler</option>
                    </select>
                  </label>
                  <Slider
                    label="Physics Freq"
                    value={config.freq}
                    min={250}
                    max={2000}
                    step={10}
                    onChange={(v) => updateConfig("freq", v)}
                  />
                  <Slider
                    label="State Ctrl Freq"
                    value={config.state_freq}
                    min={20}
                    max={200}
                    step={5}
                    onChange={(v) => updateConfig("state_freq", v)}
                  />
                </div>
              )}
            </div>

            <div className="swarm-start">
              <button
                className="primary-action"
                disabled={loading}
                onClick={() => void startSimulation()}
              >
                {loading ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
                {loading ? "Simulating..." : "Start Simulation"}
              </button>
              {error && <p className="swarm-error">{error}</p>}
            </div>

            {playback && (
              <div className="swarm-section">
                <div className="section-title">
                  <Download size={18} />
                  <h2>Export</h2>
                </div>
                <div className="swarm-export-actions">
                  <button className="secondary-action compact" onClick={() => downloadCSV(playback, config)}>
                    CSV
                  </button>
                  <button className="secondary-action compact" onClick={() => downloadJSONWaypoints(playback, config)}>
                    JSON
                  </button>
                  <button className="secondary-action compact" onClick={() => downloadROS(playback, config)}>
                    ROS
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="swarm-preview" ref={previewRef}>
            <div className="swarm-preview-toolbar">
              <span className="eyebrow">Visual</span>
              <div style={{ display: "flex", gap: 8 }}>
                {playback && (
                  <button className="secondary-action compact" onClick={() => downloadCSV(playback, config)}>
                    <Download size={14} />
                    Export
                  </button>
                )}
                <button className="secondary-action compact" onClick={() => void togglePreviewFullscreen()}>
                  {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                  {isFullscreen ? "Exit" : "Full screen"}
                </button>
              </div>
            </div>
            {loading && simPhase ? (
              <ProgressIndicator phase={simPhase.phase} percent={simPhase.percent} />
            ) : playback ? (
              <Player playback={playback} overlays={overlays ?? undefined} onClose={() => { setPlayback(null); setOverlays(null); }} autoPlay embedded loop />
            ) : (
              <div className="swarm-placeholder">
                <FlaskConical size={48} />
                <p>Adjust the parameters and start the simulation</p>
              </div>
            )}
          </div>
        </div>

        {playback && (
          <ReportPanel
            playback={playback}
            overlays={overlays ?? undefined}
            onExportCSV={() => downloadCSV(playback, config)}
            onExportJSON={() => downloadJSONWaypoints(playback, config)}
            onExportROS={() => downloadROS(playback, config)}
            onExportTXT={() => downloadReportTxt(playback, config, overlays ?? undefined)}
            onExportPDF={() => downloadReportPdf(playback, config, overlays ?? undefined)}
          />
        )}

        <div className="swarm-stats">
          <span>Drones: {playback?.numDrones ?? config.n_drones}</span>
          <span>Boundary: {config.boundary_mode}</span>
          <span>Device: {config.device}</span>
          <span>Duration: {playback ? formatTime(playback.timestamps[playback.timestamps.length - 1]) : formatTime(config.duration)}</span>
          {overlays && (
            <>
              <span>Avg Speed: {(overlays.speeds.flat().reduce((a, b) => a + b, 0) / Math.max(1, overlays.speeds.flat().length)).toFixed(2)} m/s</span>
              <span>FPS: {playback ? (playback.timestamps.length / (playback.timestamps[playback.timestamps.length - 1] - playback.timestamps[0])).toFixed(0) : "-"}</span>
              <span>Collisions: {overlays.collisions_per_frame.reduce((s, f) => s + f.length, 0)}</span>
            </>
          )}
        </div>
      </section>
    </main>
  );
}
