import { ArrowRight, Atom, BarChart3, Pyramid, Rocket, Star, User, Zap } from "lucide-react";
import { useState } from "react";
import type { SwarmConfig } from "./types";

const ONBOARDING_KEY = "dronemd_onboarding_done";

const STEPS = [
  {
    icon: <Zap size={32} />,
    title: "Welcome to DroneMD",
    body: "A real-time drone swarm simulator built for AMD GPUs. Powered by JAX and ROCm, it runs actual physics — not approximations — on the same MI300X accelerators powering frontier AI labs.",
  },
  {
    icon: <Atom size={32} />,
    title: "Physics You Can Trust",
    body: "Every drone follows Newtonian dynamics with SO(3) rotational models, rotor thrust, and aerodynamic drag. Tune separation, alignment, and cohesion weights with real-time feedback — this is swarm engineering, not animation.",
  },
  {
    icon: <BarChart3 size={32} />,
    title: "Run & Analyze",
    body: "Hit Start Simulation to deploy your swarm config to the GPU. Watch live 3D playback with collision overlays, connectivity graphs, and speed heatmaps. Scroll down for a full analytical report with charts, safety scores, and GPU benchmarks.",
  },
  {
    icon: <Rocket size={32} />,
    title: "Ship It",
    body: "Export your results as PDF reports, plain text summaries, or raw CSV / JSON / ROS waypoint files. Every run captures flight metrics, formation error, path efficiency, coverage, and GPU throughput — ready for your next design review.",
  },
];

export type DemoPreset = {
  id: string;
  label: string;
  description: string;
  icon: React.ReactNode;
  config: Partial<SwarmConfig>;
  autoStart?: boolean;
  playbackCache?: "human-body" | "star";
  objShape?: "human";
};

const DEFAULT_PRESET: DemoPreset = {
  id: "default",
  label: "Default",
  description: "25 drones in a layered rotating cone — the default demo run",
  icon: <Pyramid size={16} />,
  config: {
    n_drones: 25,
    duration: 20,
    separation_weight: 1.0,
    alignment_weight: 0.5,
    cohesion_weight: 0.5,
    perception_radius: 3.0,
    max_speed: 2.0,
    max_force: 0.5,
    boundary_mode: "wrap",
    bounds: [-2.0, 2.0, -2.0, 2.0],
    device: "cpu",
    physics: "first_principles",
    integrator: "euler",
    motion_primitive: "cone",
    primitive_params: { delta_height: 0.3, spacing: 0.5, t_form: 3.0, omega: 0.3 },
    obj_points: null,
  },
};

const CONE_PRESET: DemoPreset = {
  id: "cone",
  label: "Layered Cone",
  description: "25 drones forming a multi-layer rotating cone — showcases 3D formations",
  icon: <Pyramid size={16} />,
  config: {
    n_drones: 25,
    duration: 20,
    separation_weight: 1.0,
    alignment_weight: 0.5,
    cohesion_weight: 0.5,
    perception_radius: 3.0,
    max_speed: 2.0,
    max_force: 0.5,
    boundary_mode: "wrap",
    bounds: [-2.0, 2.0, -2.0, 2.0],
    device: "cpu",
    physics: "first_principles",
    integrator: "euler",
    motion_primitive: "cone",
    primitive_params: { delta_height: 0.3, spacing: 0.5, t_form: 3.0, omega: 0.3 },
    obj_points: null,
  },
};

const STAR_PRESET: DemoPreset = {
  id: "star",
  label: "Rotating Star",
  description: "30 drones forming a two-spoke rotating star — showcases multi-ring formations",
  icon: <Star size={16} />,
  config: {
    n_drones: 30,
    duration: 20,
    separation_weight: 1.0,
    alignment_weight: 0.5,
    cohesion_weight: 0.5,
    perception_radius: 3.0,
    max_speed: 2.0,
    max_force: 0.5,
    boundary_mode: "wrap",
    bounds: [-2.0, 2.0, -2.0, 2.0],
    device: "cpu",
    physics: "first_principles",
    integrator: "euler",
    motion_primitive: "star",
    primitive_params: { radius: 1.2, delta_radius: 0.4, t_form: 3.0, omega: 0.3 },
    obj_points: null,
  },
};

const HUMAN_BODY_PRESET: DemoPreset = {
  id: "human-body",
  label: "Human Body",
  description: "70 drones forming a human body shape — cached playback",
  icon: <User size={16} />,
  config: {
    n_drones: 70,
    duration: 10,
    device: "cpu",
  },
  autoStart: true,
  playbackCache: "human-body",
  objShape: "human",
};

const STAR_CACHED_PRESET: DemoPreset = {
  id: "star-cached",
  label: "Rotating Star",
  description: "12 drones star formation — light cached playback",
  icon: <Star size={16} />,
  config: {
    n_drones: 12,
    duration: 10,
    motion_primitive: "star",
    primitive_params: { radius: 1.0, delta_radius: 0.3, t_form: 2.0, omega: 0.2 },
    freq: 250,
    state_freq: 50,
    device: "cpu",
  },
  autoStart: true,
  playbackCache: "star",
};

export const DEMO_PRESETS: DemoPreset[] = [
  CONE_PRESET,
  HUMAN_BODY_PRESET,
  STAR_CACHED_PRESET,
];

export function isOnboardingDone(): boolean {
  return localStorage.getItem(ONBOARDING_KEY) === "1";
}

export function markOnboardingDone(): void {
  localStorage.setItem(ONBOARDING_KEY, "1");
}

type OnboardingProps = {
  onDismiss: () => void;
};

export function Onboarding({ onDismiss }: OnboardingProps) {
  const [step, setStep] = useState(0);

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  return (
    <div className="onboard-backdrop" onClick={onDismiss}>
      <div className="onboard-card" onClick={(e) => e.stopPropagation()}>
        <div className="onboard-step-icon">{current.icon}</div>
        <h2 className="onboard-step-title">{current.title}</h2>
        <p className="onboard-step-body">{current.body}</p>

        <div className="onboard-dots">
          {STEPS.map((_, i) => (
            <div key={i} className={`onboard-dot ${i === step ? "active" : ""}`} />
          ))}
        </div>

        <div className="onboard-actions">
          {step > 0 && (
            <button className="secondary-action compact" onClick={() => setStep(step - 1)}>
              Back
            </button>
          )}
          {isLast ? (
            <button className="primary-action compact" onClick={onDismiss}>
              Got it
            </button>
          ) : (
            <button className="primary-action compact" onClick={() => setStep(step + 1)}>
              Next
              <ArrowRight size={16} />
            </button>
          )}
        </div>

        <button className="onboard-skip" onClick={onDismiss}>
          Skip tour
        </button>
      </div>
    </div>
  );
}
