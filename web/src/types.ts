export type SwarmConfig = {
  n_drones: number;
  duration: number;
  separation_weight: number;
  alignment_weight: number;
  cohesion_weight: number;
  perception_radius: number;
  max_speed: number;
  max_force: number;
  boundary_mode: "wrap" | "bounce" | "hard";
  bounds: [number, number, number, number];
  obstacles: { x: number; y: number; radius: number }[];
  device: "cpu" | "gpu";
  physics: "first_principles" | "so_rpy" | "so_rpy_rotor" | "so_rpy_rotor_drag";
  integrator: "euler" | "rk4" | "symplectic_euler";
  freq: number;
  state_freq: number;
  height: number;
  motion_primitive: "none" | "circle" | "star" | "cone";
  primitive_params: Record<string, unknown>;
};

export type PlaybackOverlays = {
  collisions_per_frame: [number, number][][];
  speeds: number[][];
};

export type Playback = {
  schemaVersion: number;
  numDrones: number;
  timestamps: number[];
  states: number[][][];
  fields: {
    pos: [number, number];
    quat: [number, number];
    vel: [number, number];
    angVel: [number, number];
  };
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
  colors: number[][];
  sampleRate: number;
  overlays?: PlaybackOverlays;
};

export type SimPhase = {
  phase: string;
  percent: number;
};

export type SimReport = {
  totalCollisions: number;
  avgSpeed: number;
  maxSpeed: number;
  minSpeed: number;
  safetyScore: number;
  flightDuration: number;
  totalDistance: number;
  energyMetric: number;
  collisionTimeline: number[];
};
