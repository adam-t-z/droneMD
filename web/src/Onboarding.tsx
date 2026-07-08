import { ArrowRight, Atom, BarChart3, Cpu, Rocket, Zap } from "lucide-react";
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
};

const FLOCKING_PRESET: DemoPreset = {
  id: "flocking",
  label: "Classic Flocking",
  description: "Standard boids-like behavior at moderate scale — recommended starting point",
  icon: <Zap size={16} />,
  config: {
    n_drones: 30,
    duration: 20,
    separation_weight: 1.5,
    alignment_weight: 1.0,
    cohesion_weight: 1.0,
    perception_radius: 3.0,
    max_speed: 2.0,
    max_force: 0.5,
    boundary_mode: "wrap",
    bounds: [-2.0, 2.0, -2.0, 2.0],
    device: "rocm",
    physics: "first_principles",
    integrator: "euler",
    motion_primitive: "none",
    primitive_params: {},
    obj_points: null,
  },
};

const DENSE_PRESET: DemoPreset = {
  id: "dense",
  label: "Dense Swarm",
  description: "100 drones squeezed into a 1m box — stress test for collision avoidance",
  icon: <Atom size={16} />,
  config: {
    n_drones: 100,
    duration: 15,
    separation_weight: 3.0,
    alignment_weight: 1.5,
    cohesion_weight: 2.0,
    perception_radius: 2.0,
    max_speed: 1.5,
    max_force: 1.0,
    boundary_mode: "bounce",
    bounds: [-0.5, 0.5, -0.5, 0.5],
    device: "cpu",
    physics: "first_principles",
    integrator: "rk4",
    motion_primitive: "none",
    primitive_params: {},
    obj_points: null,
  },
};

const CIRCLE_PRESET: DemoPreset = {
  id: "circle",
  label: "Rotating Circle",
  description: "40 drones holding a spinning ring — showcases motion primitives",
  icon: <Cpu size={16} />,
  config: {
    n_drones: 40,
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
    motion_primitive: "circle",
    primitive_params: { radius: 1.5, rotation: 0.3 },
    obj_points: null,
  },
};

const SPARSE_PRESET: DemoPreset = {
  id: "sparse",
  label: "Wide Patrol",
  description: "20 drones surveying a larger 5m area with low cohesion",
  icon: <Rocket size={16} />,
  config: {
    n_drones: 20,
    duration: 30,
    separation_weight: 1.0,
    alignment_weight: 1.0,
    cohesion_weight: 0.5,
    perception_radius: 4.0,
    max_speed: 3.0,
    max_force: 0.8,
    boundary_mode: "bounce",
    bounds: [-2.5, 2.5, -2.5, 2.5],
    device: "cpu",
    physics: "first_principles",
    integrator: "rk4",
    motion_primitive: "none",
    primitive_params: {},
    obj_points: null,
  },
};

export const DEMO_PRESETS: DemoPreset[] = [
  FLOCKING_PRESET,
  DENSE_PRESET,
  CIRCLE_PRESET,
  SPARSE_PRESET,
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
