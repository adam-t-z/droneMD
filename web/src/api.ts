import type { BenchmarkHistory, Playback, SimPhase, SwarmConfig } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    let detail: string | null = null;
    try {
      const payload = JSON.parse(text) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      }
    } catch {
      detail = null;
    }
    if (detail) {
      throw new Error(detail);
    }
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function loadDefaultPlayback(): Promise<Playback | null> {
  try {
    const resp = await fetch("/data/default-playback.json");
    if (!resp.ok) return null;
    return (await resp.json()) as Playback;
  } catch {
    return null;
  }
}

export function simulateSwarm(config: SwarmConfig): Promise<Playback> {
  return request<Playback>("/api/swarm/simulate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function simulateSwarmStream(
  config: SwarmConfig,
  onProgress: (phase: SimPhase) => void,
): Promise<Playback> {
  const response = await fetch("/api/swarm/simulate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const msg = JSON.parse(line) as { type: string; phase?: string; percent?: number; data?: Playback; message?: string };
      if (msg.type === "progress" && msg.phase !== undefined && msg.percent !== undefined) {
        onProgress({ phase: msg.phase, percent: msg.percent });
      } else if (msg.type === "result" && msg.data) {
        return msg.data;
      } else if (msg.type === "error") {
        throw new Error(msg.message ?? "Simulation stream error");
      }
    }
  }

  throw new Error("Stream ended without result");
}

export async function fetchDefaultObjPoints(): Promise<{ points: number[][]; n: number } | null> {
  try {
    const resp = await fetch("/data/default-obj-points.json");
    if (!resp.ok) return null;
    return await resp.json() as { points: number[][]; n: number };
  } catch {
    return null;
  }
}

export async function uploadObjFile(
  file: File,
  nDrones: number,
): Promise<{ points: number[][]; n_drones: number }> {
  const formData = new FormData();
  formData.append("file", file);
  const resp = await fetch(`/api/swarm/shapes/upload-obj?n_drones=${nDrones}`, {
    method: "POST",
    body: formData,
  });
  if (!resp.ok) {
    const text = await resp.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text) as { detail?: string };
      if (parsed.detail) detail = parsed.detail;
    } catch { /* use raw text */ }
    throw new Error(detail);
  }
  return await resp.json() as { points: number[][]; n_drones: number };
}

export function fetchGpuBenchmark(): Promise<BenchmarkHistory> {
  return request<BenchmarkHistory>("/api/swarm/benchmark");
}
