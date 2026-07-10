import { Atom, BookOpen, ChevronDown, ChevronLeft, ChevronRight, FlaskConical, Loader2, Maximize2, Minimize2, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchDefaultObjPoints, loadDefaultPlayback, loadHumanBodyPlayback, uploadObjFile } from "./api";
import { downloadCSV, downloadJSONWaypoints, downloadROS, downloadReportPdf, downloadReportTxt } from "./export";
import { DEMO_PRESETS, isOnboardingDone, markOnboardingDone, Onboarding } from "./Onboarding";
import type { DemoPreset } from "./Onboarding";
import { Player } from "./Player";
import { ReportTabs } from "./ReportTabs";
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
  motion_primitive: "cone",
  primitive_params: { delta_height: 0.3, spacing: 0.5, t_form: 3.0 },
  obj_points: null,
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
  description?: string;
  min: number;
  max: number;
  step: number;
};

type SelectOption = { value: string; label: string };

type PrimitiveSliderDef = {
  paramKey: string;
  label: string;
  description: string;
  min: number;
  max: number;
  step: number;
  defaultVal: number;
};

const DRONE_SLIDERS: SliderDef[] = [
  { key: "n_drones", label: "Drone Count", description: "Number of drones in the simulation.\nMore drones create denser swarm behavior,\nbut increase computation time.", min: 10, max: 200, step: 1 },
  { key: "duration", label: "Duration (s)", description: "Total simulation time in seconds.\nLonger simulations show more developed\nswarming patterns.", min: 5, max: 120, step: 1 },
];

const BEHAVIOR_SLIDERS: SliderDef[] = [
  { key: "separation_weight", label: "Separation", description: "How strongly drones avoid neighbors.\n0 = ignore\n5 = aggressively avoid", min: 0, max: 5, step: 0.1 },
  { key: "alignment_weight", label: "Alignment", description: "How strongly drones match neighbors'\ndirection.\n0 = ignore\n5 = strictly align", min: 0, max: 5, step: 0.1 },
  { key: "cohesion_weight", label: "Cohesion", description: "How strongly drones move toward\nthe group center.\n0 = ignore\n5 = tightly cluster", min: 0, max: 5, step: 0.1 },
  { key: "perception_radius", label: "Perception Radius", description: "How far a drone can sense neighbors.\nLarger values create more connected\nswarms but reduce local responsiveness.", min: 0.5, max: 10, step: 0.1 },
  { key: "max_speed", label: "Max Speed", description: "Maximum drone speed (m/s).\nHigher speeds enable faster movement\nbut can reduce stability.", min: 0.5, max: 5, step: 0.1 },
  { key: "max_force", label: "Max Force", description: "Maximum steering acceleration.\nHigher values make drones more\nresponsive but can cause jerky motion.", min: 0.1, max: 3, step: 0.1 },
];

const FORMATION_DESC = "None: drones spawn randomly\nHuman Body: spawn in a preset human shape\nUpload OBJ: spawn using a custom 3D model";

const PRIMITIVE_TYPE_DESC = "None: standard flocking behavior\nCircle: drones form a rotating ring\nStar: drones form a rotating star shape\nCone: drones stack into a 3D cone";

const BOUNDARY_MODE_DESC = "How drones behave at the boundary.\nWrap: teleport to opposite side\nBounce: reflect off the wall\nHard: stop at the wall";

const DEVICE_DESC = "CPU: runs on processor (works everywhere)\nGPU (CUDA): NVIDIA GPU via CUDA\nGPU (ROCm): AMD GPU via ROCm";

const PHYSICS_DESC = "First Principles: basic Newtonian physics\nSO(3) RPY: full 3D rotational dynamics\nSO(3) RPY Rotor: adds rotor thrust model\nSO(3) RPY Rotor Drag: adds aerodynamic drag";

const INTEGRATOR_DESC = "Euler: simplest, least accurate, fastest\nRK4: most accurate, slower\nSymplectic Euler: good energy conservation\nwith balanced accuracy/speed";

const ADVANCED_SLIDERS: SliderDef[] = [
  { key: "freq", label: "Physics Freq", description: "Physics simulation steps per second.\nHigher values improve accuracy\nbut increase computation time.", min: 250, max: 2000, step: 10 },
  { key: "state_freq", label: "State Ctrl Freq", description: "How often the drone controller updates\nper second. Higher values give\nmore precise control.", min: 20, max: 200, step: 5 },
];

const CIRCLE_PARAMS: PrimitiveSliderDef[] = [
  { paramKey: "radius", label: "Radius", description: "Radius of the circular formation.\nLarger circles spread drones farther apart.", min: 0.3, max: 2.0, step: 0.1, defaultVal: 1.5 },
  { paramKey: "rotation", label: "Rotation (rad/s)", description: "How fast the circle rotates.\nSet to 0 for a static formation.", min: 0, max: 1.5, step: 0.1, defaultVal: 0.3 },
];

const STAR_PARAMS: PrimitiveSliderDef[] = [
  { paramKey: "radius", label: "Inner Radius", description: "Distance from center to the inner\nvertices. Smaller values create more\npronounced star points.", min: 0.3, max: 2.0, step: 0.1, defaultVal: 1.2 },
  { paramKey: "delta_radius", label: "Spoke Gap", description: "How far the outer vertices extend\nbeyond the inner ring.", min: 0.1, max: 1.0, step: 0.05, defaultVal: 0.4 },
  { paramKey: "rotation", label: "Rotation (rad/s)", description: "How fast the star rotates.\nSet to 0 for a static formation.", min: 0, max: 1.5, step: 0.1, defaultVal: 0.2 },
];

const CONE_PARAMS: PrimitiveSliderDef[] = [
  { paramKey: "delta_height", label: "Layer Height", description: "Vertical spacing between each ring\nlayer. Smaller values create a denser cone.", min: 0.1, max: 0.8, step: 0.05, defaultVal: 0.3 },
  { paramKey: "spacing", label: "Spacing", description: "Horizontal spacing between drones\nwithin each ring.", min: 0.3, max: 1.2, step: 0.05, defaultVal: 0.5 },
  { paramKey: "rotation", label: "Rotation (rad/s)", description: "How fast the cone rotates.\nSet to 0 for a static formation.", min: 0, max: 1.5, step: 0.1, defaultVal: 0.3 },
];

const CONE_INVERTED_DESC = "When enabled, the cone apex points\ndown instead of up.";

function ParamTooltip({ description }: { description: string }) {
  const lines = description.split("\n");
  return (
    <span className="param-tooltip-wrap">
      <span className="param-tooltip-icon">?</span>
      <span className="param-tooltip-body">
        {lines.map((line, i) => [
          i > 0 && <br key={`br-${i}`} />,
          <span key={i}>{line}</span>,
        ])}
      </span>
    </span>
  );
}

function Slider({
  label,
  description,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  description?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="swarm-slider">
      <span className="param-label-row">
        {label}
        {description && <ParamTooltip description={description} />}
      </span>
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

function SelectControl({
  label,
  description,
  value,
  options,
  onChange,
}: {
  label: string;
  description?: string;
  value: string;
  options: SelectOption[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="swarm-slider">
      <span className="param-label-row">
        {label}
        {description && <ParamTooltip description={description} />}
      </span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
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
  const cachedPlaybackRef = useRef<Playback | null>(null);
  const cachedOverlaysRef = useRef<PlaybackOverlays | null>(null);
  const humanBodyCacheRef = useRef<{ playback: Playback; overlays: PlaybackOverlays | null } | null>(null);
  const [showBehavior, setShowBehavior] = useState(false);
  const [showEnvironment, setShowEnvironment] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  const [formationType, setFormationType] = useState<"none" | "human" | "upload">("none");
  const [objUploading, setObjUploading] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(() => !isOnboardingDone());
  const previewRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleFormationChange = useCallback(async (value: "none" | "human" | "upload") => {
    setFormationType(value);
    if (value === "none") {
      setConfig((prev) => ({ ...prev, obj_points: null }));
      return;
    }
    if (value === "human") {
      const data = await fetchDefaultObjPoints();
      if (data) {
        setConfig((prev) => ({ ...prev, n_drones: data.n, obj_points: data.points }));
      }
    }
    if (value === "upload") {
      setConfig((prev) => ({ ...prev, obj_points: null }));
      setTimeout(() => fileInputRef.current?.click(), 50);
    }
  }, []);

  const handleObjFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setObjUploading(true);
      try {
        const result = await uploadObjFile(file, config.n_drones);
        setConfig((prev) => ({
          ...prev,
          obj_points: result.points,
          n_drones: result.n_drones,
        }));
      } catch (err) {
        setError(err instanceof Error ? err.message : "OBJ upload failed");
        setFormationType("none");
        setConfig((prev) => ({ ...prev, obj_points: null }));
      } finally {
        setObjUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [config.n_drones],
  );

  const startSimulation = useCallback(async () => {
    setError(null);
    setLoading(true);
    setSimPhase({ phase: "Initializing simulation engine", percent: 0 });
    try {
      for (let i = 0; i < PHASES.length; i++) {
        const pct = Math.round(((i + 1) / PHASES.length) * 100);
        setSimPhase({ phase: PHASES[i].key, percent: pct });
        await new Promise((r) => setTimeout(r, 500));
      }
      if (config.obj_points && humanBodyCacheRef.current) {
        setPlayback(humanBodyCacheRef.current.playback);
        setOverlays(humanBodyCacheRef.current.overlays);
      } else {
        if (!cachedPlaybackRef.current) {
          const data = await loadDefaultPlayback();
          if (!data) throw new Error("No default playback data available");
          const { overlays: ov, ...rest } = data;
          cachedPlaybackRef.current = rest;
          cachedOverlaysRef.current = ov ?? null;
        }
        setPlayback(cachedPlaybackRef.current);
        setOverlays(cachedOverlaysRef.current);
      }
      setShowReport(true);
      setSidebarCollapsed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      setLoading(false);
      setSimPhase(null);
    }
  }, [config.obj_points]);

  const reset = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
    setError(null);
    setPlayback(null);
    setOverlays(null);
    setSimPhase(null);
    setShowReport(false);
    setFormationType("none");
    setActivePresetId(null);
    setSidebarCollapsed(false);
  }, []);

  const loadPreset = useCallback((preset: DemoPreset) => {
    const merged = { ...DEFAULT_CONFIG, ...preset.config };
    setConfig(merged);
    setActivePresetId(preset.id);
    setError(null);
    setPlayback(null);
    setOverlays(null);
    setSimPhase(null);
    setShowReport(false);
    setSidebarCollapsed(false);
    if (preset.config.motion_primitive) {
      setFormationType("none");
    }
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

  useEffect(() => {
    void loadDefaultPlayback().then((data) => {
      if (data) {
        const { overlays: ov, ...rest } = data;
        cachedPlaybackRef.current = rest;
        cachedOverlaysRef.current = ov ?? null;
        setPlayback(rest);
        setOverlays(ov ?? null);
        setConfig({ ...DEFAULT_CONFIG, ...DEMO_PRESETS[0].config });
        setActivePresetId("default");
      }
    });
    void loadHumanBodyPlayback().then((data) => {
      if (data) {
        const { overlays: ov, ...rest } = data;
        humanBodyCacheRef.current = { playback: rest, overlays: ov ?? null };
      }
    });
  }, []);

  const statusLabel = loading ? "Simulating" : playback ? "Complete" : "Ready";
  const statusClass = loading
    ? "status-pill playing"
    : playback
    ? "status-pill playing"
    : "status-pill";

  return (
    <main className="app-shell swarm-lab">
      {showOnboarding && (
        <Onboarding
          onDismiss={() => {
            markOnboardingDone();
            setShowOnboarding(false);
          }}
        />
      )}
      <section className="workspace">
        <header className="topbar">
          <div>
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
            <span className="meta-chip demo-chip">Demo — Cached Results</span>
            <button
              className="primary-action"
              disabled={loading}
              onClick={() => void startSimulation()}
            >
              {loading ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
              {loading ? "Simulating..." : "Start Simulation"}
            </button>
            <button className="secondary-action compact" onClick={reset}>
              <RotateCcw size={18} />
              Reset
            </button>
            {error && <p className="swarm-error">{error}</p>}
          </div>
        </header>

        <div className={`swarm-layout${showReport ? " has-results" : ""}${showReport && sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
          <div className="swarm-controls">
            {playback && (
              <div className="swarm-controls-collapsed-bar">
                <button
                  className="swarm-controls-expand-btn"
                  onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                  title={sidebarCollapsed ? "Expand controls" : "Collapse controls"}
                >
                  {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
                </button>
                <span className="swarm-controls-collapsed-label">Params</span>
                <button className="secondary-action compact" onClick={reset} title="Reset">
                  <RotateCcw size={16} />
                </button>
              </div>
            )}
            <div className="preset-strip">
              {DEMO_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  className={`preset-chip${activePresetId === preset.id ? " recommended" : ""}`}
                  onClick={() => loadPreset(preset)}
                  title={preset.description}
                >
                  {preset.icon}
                  {preset.label}
                </button>
              ))}
              <button
                className="preset-chip"
                onClick={() => { markOnboardingDone(); setShowOnboarding(true); }}
                title="Replay the tour"
              >
                <BookOpen size={14} />
                Tour
              </button>
            </div>
            <div className="swarm-section">
              <div className="section-title">
                <Atom size={18} />
                <h2>Drones</h2>
              </div>
              {DRONE_SLIDERS.map((s) => (
                <Slider
                  key={s.key}
                  label={s.label}
                  description={s.description}
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
                <h2>Drone Formation</h2>
              </div>
              <SelectControl
                label="Shape"
                description={FORMATION_DESC}
                value={formationType}
                options={[
                  { value: "none", label: "None" },
                  { value: "human", label: "Human Body" },
                  { value: "upload", label: "Upload OBJ..." },
                ]}
                onChange={(v) => void handleFormationChange(v as "none" | "human" | "upload")}
              />
              {formationType === "upload" && (
                <div className="swarm-obj-upload">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".obj"
                    style={{ display: "none" }}
                    onChange={(e) => void handleObjFileChange(e)}
                  />
                  <button
                    className="secondary-action compact"
                    disabled={objUploading}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    {objUploading ? "Processing..." : "Choose OBJ File"}
                  </button>
                  {config.obj_points && (
                    <span className="swarm-obj-status">
                      {config.obj_points.length} points loaded
                    </span>
                  )}
                  <p className="spawn-csv-hint">
                    Upload a .obj 3D model. Drones will spawn in the model's shape and hold position.
                  </p>
                </div>
              )}
              {formationType === "human" && config.obj_points && (
                <p className="spawn-csv-hint">
                  {config.obj_points.length} drones will form the human body shape and hold position.
                </p>
              )}
            </div>

            <div className="swarm-section">
              <div className="section-title">
                <h2>Motion Primitive</h2>
              </div>
              <SelectControl
                label="Shape"
                description={PRIMITIVE_TYPE_DESC}
                value={config.motion_primitive}
                options={[
                  { value: "none", label: "None (Flocking)" },
                  { value: "circle", label: "Circle" },
                  { value: "star", label: "Star" },
                  { value: "cone", label: "Cone" },
                ]}
                onChange={(v) => updateConfig("motion_primitive", v as SwarmConfig["motion_primitive"])}
              />
              {config.motion_primitive === "circle" && (
                <>
                  {CIRCLE_PARAMS.map((p) => (
                    <Slider
                      key={p.paramKey}
                      label={p.label}
                      description={p.description}
                      value={getPrimitiveParam(p.paramKey, p.defaultVal)}
                      min={p.min}
                      max={p.max}
                      step={p.step}
                      onChange={(v) => updatePrimitiveParam(p.paramKey, v)}
                    />
                  ))}
                  <p className="spawn-csv-hint">
                    Drones form a circle and hold the shape continuously (rotate at 0 to hold still).
                  </p>
                </>
              )}
              {config.motion_primitive === "star" && (
                <>
                  {STAR_PARAMS.map((p) => (
                    <Slider
                      key={p.paramKey}
                      label={p.label}
                      description={p.description}
                      value={getPrimitiveParam(p.paramKey, p.defaultVal)}
                      min={p.min}
                      max={p.max}
                      step={p.step}
                      onChange={(v) => updatePrimitiveParam(p.paramKey, v)}
                    />
                  ))}
                  <p className="spawn-csv-hint">
                    Two interleaved rings form a star with n/2 spokes; shape is held continuously.
                  </p>
                </>
              )}
              {config.motion_primitive === "cone" && (
                <>
                  {CONE_PARAMS.map((p) => (
                    <Slider
                      key={p.paramKey}
                      label={p.label}
                      description={p.description}
                      value={getPrimitiveParam(p.paramKey, p.defaultVal)}
                      min={p.min}
                      max={p.max}
                      step={p.step}
                      onChange={(v) => updatePrimitiveParam(p.paramKey, v)}
                    />
                  ))}
                  <label className="swarm-slider">
                    <span className="param-label-row">
                      Inverted
                      <ParamTooltip description={CONE_INVERTED_DESC} />
                    </span>
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
                <h2>Behavior</h2>
              </button>
              {showBehavior && (
                <div className="swarm-advanced-body">
                  {BEHAVIOR_SLIDERS.map((s) => (
                    <Slider
                      key={s.key}
                      label={s.label}
                      description={s.description}
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
                  <SelectControl
                    label="Boundary Mode"
                    description={BOUNDARY_MODE_DESC}
                    value={config.boundary_mode}
                    options={[
                      { value: "wrap", label: "Wrap" },
                      { value: "bounce", label: "Bounce" },
                      { value: "hard", label: "Hard" },
                    ]}
                    onChange={(v) => updateConfig("boundary_mode", v as SwarmConfig["boundary_mode"])}
                  />
                  <SelectControl
                    label="Device"
                    description={DEVICE_DESC}
                    value={config.device}
                    options={[
                      { value: "cpu", label: "CPU" },
                      { value: "cuda", label: "GPU (CUDA / NVIDIA)" },
                      { value: "rocm", label: "GPU (ROCm / AMD)" },
                    ]}
                    onChange={(v) => updateConfig("device", v as SwarmConfig["device"])}
                  />
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
                  <SelectControl
                    label="Physics Model"
                    description={PHYSICS_DESC}
                    value={config.physics}
                    options={[
                      { value: "first_principles", label: "First Principles" },
                      { value: "so_rpy", label: "SO(3) RPY" },
                      { value: "so_rpy_rotor", label: "SO(3) RPY Rotor" },
                      { value: "so_rpy_rotor_drag", label: "SO(3) RPY Rotor Drag" },
                    ]}
                    onChange={(v) => updateConfig("physics", v as SwarmConfig["physics"])}
                  />
                  <SelectControl
                    label="Integrator"
                    description={INTEGRATOR_DESC}
                    value={config.integrator}
                    options={[
                      { value: "euler", label: "Euler" },
                      { value: "rk4", label: "RK4" },
                      { value: "symplectic_euler", label: "Symplectic Euler" },
                    ]}
                    onChange={(v) => updateConfig("integrator", v as SwarmConfig["integrator"])}
                  />
                  {ADVANCED_SLIDERS.map((s) => (
                    <Slider
                      key={s.key}
                      label={s.label}
                      description={s.description}
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

          </div>

          <div className="swarm-preview" ref={previewRef}>
            <div className="swarm-preview-toolbar">
              <span className="eyebrow">Visual</span>
              <button className="secondary-action compact" onClick={() => void togglePreviewFullscreen()}>
                {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                {isFullscreen ? "Exit" : "Full screen"}
              </button>
            </div>
            {loading && simPhase ? (
              <ProgressIndicator phase={simPhase.phase} percent={simPhase.percent} />
            ) : playback ? (
              <Player playback={playback} overlays={overlays ?? undefined} onClose={() => { setPlayback(null); setOverlays(null); setShowReport(false); setSidebarCollapsed(false); }} autoPlay embedded loop />
            ) : (
              <div className="swarm-placeholder">
                <FlaskConical size={48} />
                <p>Adjust the parameters and start the simulation</p>
              </div>
            )}
          </div>

          {showReport && playback && (
            <ReportTabs
              playback={playback}
              overlays={overlays ?? undefined}
              onExportCSV={() => downloadCSV(playback, config)}
              onExportJSON={() => downloadJSONWaypoints(playback, config)}
              onExportROS={() => downloadROS(playback, config)}
              onExportTXT={() => downloadReportTxt(playback, config, overlays ?? undefined)}
              onExportPDF={() => downloadReportPdf(playback, config, overlays ?? undefined)}
            />
          )}
        </div>


      </section>
    </main>
  );
}
