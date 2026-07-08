import type { Playback, PlaybackOverlays, SimReport, SwarmConfig } from "./types";

const NEAR_MISS_THRESHOLD = 0.6; // meters, above collision threshold (0.3m)
const COLLISION_THRESHOLD = 0.3;
const CONNECTIVITY_RADIUS = 3.0;
const COVERAGE_GRID_SIZE = 30;
const MAX_SAMPLE_FRAMES = 500;

function download(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function sampleFrames<T>(data: T[], maxFrames: number): { values: T[]; indices: number[] } {
  const n = data.length;
  if (n <= maxFrames) {
    return { values: data, indices: data.map((_, i) => i) };
  }
  const step = n / maxFrames;
  const values: T[] = [];
  const indices: number[] = [];
  for (let i = 0; i < maxFrames; i++) {
    const idx = Math.floor(i * step);
    values.push(data[idx]);
    indices.push(idx);
  }
  return { values, indices };
}

export function computeReport(playback: Playback, overlays?: PlaybackOverlays): SimReport {
  const { timestamps, states, numDrones, bounds } = playback;
  const numFrames = timestamps.length;

  // --- Existing metrics ---
  let totalCollisions = 0;
  const collisionTimeline: number[] = new Array(numFrames).fill(0);
  if (overlays) {
    for (let t = 0; t < overlays.collisions_per_frame.length; t++) {
      const c = overlays.collisions_per_frame[t].length;
      totalCollisions += c;
      collisionTimeline[t] = c;
    }
  }

  let avgSpeed = 0;
  let maxSpeed = 0;
  let minSpeed = Infinity;
  let totalSamples = 0;
  if (overlays) {
    for (const row of overlays.speeds) {
      for (const s of row) {
        avgSpeed += s;
        totalSamples++;
        if (s > maxSpeed) maxSpeed = s;
        if (s < minSpeed) minSpeed = s;
      }
    }
    avgSpeed = totalSamples > 0 ? avgSpeed / totalSamples : 0;
  }
  if (minSpeed === Infinity) minSpeed = 0;

  const flightDuration = timestamps.length > 1 ? timestamps[timestamps.length - 1] - timestamps[0] : 0;
  const safetyScore = numFrames > 0
    ? ((numFrames - collisionTimeline.filter((c) => c > 0).length) / numFrames) * 100
    : 100;

  let totalDistance = 0;
  if (numFrames > 1) {
    for (let d = 0; d < numDrones; d++) {
      for (let t = 1; t < numFrames; t++) {
        const dx = states[t][d][0] - states[t - 1][d][0];
        const dy = states[t][d][1] - states[t - 1][d][1];
        const dz = states[t][d][2] - states[t - 1][d][2];
        totalDistance += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
    }
  }

  let energyMetric = 0;
  if (overlays) {
    let sqSum = 0;
    let count = 0;
    for (const row of overlays.speeds) {
      for (const s of row) {
        sqSum += s * s;
        count++;
      }
    }
    energyMetric = count > 0 ? sqSum / count : 0;
  }

  // --- New metrics ---

  // Speed timeline: mean speed across all drones per frame
  const speedTimeline: number[] = new Array(numFrames).fill(0);
  for (let t = 0; t < numFrames; t++) {
    let sum = 0;
    for (let d = 0; d < numDrones; d++) {
      sum += Math.sqrt(
        states[t][d][7] * states[t][d][7] +
        states[t][d][8] * states[t][d][8] +
        states[t][d][9] * states[t][d][9],
      );
    }
    speedTimeline[t] = sum / numDrones;
  }

  // Pairwise metrics: downsampled to MAX_SAMPLE_FRAMES
  const sampled = sampleFrames(states as number[][][], MAX_SAMPLE_FRAMES);
  const sampledStates = sampled.values;
  const sampledIndices = sampled.indices;
  const numSampled = sampledStates.length;

  // Pre-allocate per-drone path accumulators (for path efficiency)
  const dronePaths: number[] = new Array(numDrones).fill(0);
  const droneStraight: { startX: number; startY: number; startZ: number }[] = [];
  const droneEnd: { x: number; y: number; z: number }[] = [];

  // Coverage grid
  const bx = bounds?.min?.[0] ?? 0;
  const by = bounds?.min?.[1] ?? 0;
  const bW = (bounds?.max?.[0] ?? 0) - bx;
  const bH = (bounds?.max?.[1] ?? 0) - by;
  const cellW = bW / COVERAGE_GRID_SIZE;
  const cellH = bH / COVERAGE_GRID_SIZE;
  const coverageGrid = new Uint8Array(COVERAGE_GRID_SIZE * COVERAGE_GRID_SIZE);

  // Compute per-drone path length + coverage (full resolution)
  if (numFrames > 1) {
    for (let d = 0; d < numDrones; d++) {
      let pathLen = 0;
      for (let t = 1; t < numFrames; t++) {
        const dx = states[t][d][0] - states[t - 1][d][0];
        const dy = states[t][d][1] - states[t - 1][d][1];
        const dz = states[t][d][2] - states[t - 1][d][2];
        pathLen += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      dronePaths[d] = pathLen;
      droneStraight.push({
        startX: states[0][d][0],
        startY: states[0][d][1],
        startZ: states[0][d][2],
      });
      droneEnd.push({
        x: states[numFrames - 1][d][0],
        y: states[numFrames - 1][d][1],
        z: states[numFrames - 1][d][2],
      });
      // Coverage: mark every Nth frame to keep performance
      const covStep = Math.max(1, Math.floor(numFrames / 200));
      for (let t = 0; t < numFrames; t += covStep) {
        const cx = Math.floor((states[t][d][0] - bx) / cellW);
        const cy = Math.floor((states[t][d][1] - by) / cellH);
        if (cx >= 0 && cx < COVERAGE_GRID_SIZE && cy >= 0 && cy < COVERAGE_GRID_SIZE) {
          coverageGrid[cy * COVERAGE_GRID_SIZE + cx] = 1;
        }
      }
    }
  }

  // Pairwise-dependent metrics on downsampled frames
  let globalMinDist = Infinity;
  let totalNearMisses = 0;
  const nearMissTimeline: number[] = new Array(numSampled).fill(0);
  const connectivityTimeline: number[] = new Array(numSampled).fill(0);
  const totalPairs = (numDrones * (numDrones - 1)) / 2;

  for (let si = 0; si < numSampled; si++) {
    let frameMinDist = Infinity;
    let frameNearMisses = 0;
    let frameConnected = 0;

    for (let i = 0; i < numDrones; i++) {
      const xi = sampledStates[si][i][0];
      const yi = sampledStates[si][i][1];
      const zi = sampledStates[si][i][2];
      for (let j = i + 1; j < numDrones; j++) {
        const dx = xi - sampledStates[si][j][0];
        const dy = yi - sampledStates[si][j][1];
        const dz = zi - sampledStates[si][j][2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < frameMinDist) frameMinDist = dist;
        if (dist > COLLISION_THRESHOLD && dist < NEAR_MISS_THRESHOLD) {
          frameNearMisses++;
        }
        if (dist < CONNECTIVITY_RADIUS) {
          frameConnected++;
        }
      }
    }
    if (frameMinDist < globalMinDist) globalMinDist = frameMinDist;
    totalNearMisses += frameNearMisses;
    nearMissTimeline[si] = frameNearMisses;
    connectivityTimeline[si] = totalPairs > 0 ? (frameConnected / totalPairs) * 100 : 0;
  }
  if (globalMinDist === Infinity) globalMinDist = 0;

  // Formation error: full resolution (O(N*T) per frame, not O(N²))
  const formationErrorTimeline = new Array(numFrames).fill(0);
  let totalFormationError = 0;
  for (let t = 0; t < numFrames; t++) {
    let cx = 0, cy = 0, cz = 0;
    for (let d = 0; d < numDrones; d++) {
      cx += states[t][d][0];
      cy += states[t][d][1];
      cz += states[t][d][2];
    }
    cx /= numDrones;
    cy /= numDrones;
    cz /= numDrones;
    let errorSum = 0;
    for (let d = 0; d < numDrones; d++) {
      const dx = states[t][d][0] - cx;
      const dy = states[t][d][1] - cy;
      const dz = states[t][d][2] - cz;
      errorSum += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    const frameError = errorSum / numDrones;
    formationErrorTimeline[t] = frameError;
    totalFormationError += frameError;
  }
  const formationError = numFrames > 0 ? totalFormationError / numFrames : 0;

  // Path efficiency per drone
  const pathEfficiencyPerDrone: number[] = new Array(numDrones).fill(0);
  let totalPathEfficiency = 0;
  for (let d = 0; d < numDrones; d++) {
    if (dronePaths[d] > 0.001) {
      const dx = droneEnd[d].x - droneStraight[d].startX;
      const dy = droneEnd[d].y - droneStraight[d].startY;
      const dz = droneEnd[d].z - droneStraight[d].startZ;
      const straightDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const eff = straightDist / dronePaths[d];
      pathEfficiencyPerDrone[d] = eff;
      totalPathEfficiency += eff;
    }
  }
  const pathEfficiency = numDrones > 0 ? totalPathEfficiency / numDrones : 0;

  // Coverage: count visited cells
  let visitedCells = 0;
  for (let i = 0; i < coverageGrid.length; i++) {
    if (coverageGrid[i] === 1) visitedCells++;
  }
  const coveragePercent = ((visitedCells / coverageGrid.length) * 100);

  // Average connectivity density across sampled frames
  const avgConnectivity = numSampled > 0
    ? connectivityTimeline.reduce((a, b) => a + b, 0) / numSampled
    : 0;

  return {
    totalCollisions,
    avgSpeed: Math.round(avgSpeed * 100) / 100,
    maxSpeed: Math.round(maxSpeed * 100) / 100,
    minSpeed: Math.round(minSpeed * 100) / 100,
    safetyScore: Math.round(safetyScore),
    flightDuration,
    totalDistance: Math.round(totalDistance * 100) / 100,
    energyMetric: Math.round(energyMetric * 100) / 100,
    collisionTimeline,
    minDistance: Math.round(globalMinDist * 1000) / 1000,
    nearMisses: totalNearMisses,
    nearMissTimeline,
    formationError: Math.round(formationError * 1000) / 1000,
    formationErrorTimeline,
    pathEfficiency: Math.round(pathEfficiency * 10000) / 10000,
    pathEfficiencyPerDrone: pathEfficiencyPerDrone.map((v) => Math.round(v * 10000) / 10000),
    coveragePercent: Math.round(coveragePercent * 10) / 10,
    connectivityDensity: Math.round(avgConnectivity * 10) / 10,
    connectivityTimeline: connectivityTimeline.map((v) => Math.round(v * 10) / 10),
    speedTimeline: speedTimeline.map((v) => Math.round(v * 100) / 100),
  };
}

function downsampleArray(data: number[], maxPoints: number): number[] {
  const n = data.length;
  if (n <= maxPoints) return data;
  const step = n / maxPoints;
  const result: number[] = [];
  for (let i = 0; i < maxPoints; i++) {
    result.push(data[Math.floor(i * step)]);
  }
  return result;
}

function svgLineChart(
  data: number[],
  width: number,
  height: number,
  color: string,
  yLabel: string,
  maxPoints = 300,
): string {
  const values = downsampleArray(data, maxPoints);
  const padL = 44, padR = 12, padT = 16, padB = 28;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const n = values.length;
  if (n < 2) return "";

  const maxVal = Math.max(...values, 0.001);
  const minVal = Math.min(...values, 0);

  const toX = (i: number) => padL + (i / (n - 1)) * w;
  const toY = (v: number) => padT + h - ((v - minVal) / (maxVal - minVal || 1)) * h;

  const points = values.map((v, i) => `${toX(i)},${toY(v)}`).join(" ");
  const areaPoints = `${toX(0)},${padT + h} ` + values.map((v, i) => `${toX(i)},${toY(v)}`).join(" ") + ` ${toX(n - 1)},${padT + h}`;

  const yTicks = 4;
  let yTickSvg = "";
  for (let i = 0; i <= yTicks; i++) {
    const val = minVal + (maxVal - minVal) * (i / yTicks);
    const y = padT + h - (i / yTicks) * h;
    yTickSvg += `<line x1="${padL - 4}" y1="${y}" x2="${padL}" y2="${y}" stroke="#cbd5e1" stroke-width="1"/><text x="${padL - 6}" y="${y + 4}" text-anchor="end" font-size="9" fill="#94a3b8">${val.toFixed(val >= 100 ? 0 : val >= 10 ? 1 : 2)}</text>`;
  }

  let xTickSvg = "";
  const xTickCount = Math.min(n, 6);
  for (let i = 0; i < xTickCount; i++) {
    const idx = Math.floor(i * (n - 1) / (xTickCount - 1 || 1));
    const x = toX(idx);
    xTickSvg += `<text x="${x}" y="${padT + h + 16}" text-anchor="middle" font-size="9" fill="#94a3b8">${idx}</text>`;
  }

  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="font-family:Inter,system-ui,sans-serif">
    <rect x="${padL}" y="${padT}" width="${w}" height="${h}" fill="#fafbfc" rx="4"/>
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + h}" stroke="#e2e8f0" stroke-width="1"/>
    <line x1="${padL}" y1="${padT + h}" x2="${padL + w}" y2="${padT + h}" stroke="#e2e8f0" stroke-width="1"/>
    ${yTickSvg}
    ${xTickSvg}
    <polygon points="${areaPoints}" fill="${color}" fill-opacity="0.12"/>
    <polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
    <text x="${padL + w / 2}" y="${height - 2}" text-anchor="middle" font-size="9" fill="#94a3b8">frame</text>
    <text x="10" y="${padT + h / 2}" text-anchor="middle" font-size="9" fill="#94a3b8" transform="rotate(-90,10,${padT + h / 2})">${yLabel}</text>
  </svg>`;
}

function svgBarChart(
  data: number[],
  labels: string[],
  width: number,
  height: number,
  color: string,
  yLabel: string,
): string {
  const values = data;
  const padL = 44, padR = 12, padT = 16, padB = 28;
  const w = width - padL - padR;
  const h = height - padT - padB;
  const n = values.length;
  if (n === 0) return "";

  const maxVal = Math.max(...values, 1);
  const barW = Math.max(1, (w / n) * 0.7);
  const gap = w / n;

  const toY = (v: number) => padT + h - (v / maxVal) * h;

  let bars = "";
  for (let i = 0; i < n; i++) {
    const bx = padL + i * gap + gap * 0.5 - barW / 2;
    const by = toY(values[i]);
    const bh = padT + h - by;
    bars += `<rect x="${bx}" y="${by}" width="${barW}" height="${Math.max(1, bh)}" fill="${color}" rx="1"/>`;
  }

  const yTicks = 4;
  let yTickSvg = "";
  for (let i = 0; i <= yTicks; i++) {
    const val = (maxVal * i) / yTicks;
    const y = padT + h - (i / yTicks) * h;
    yTickSvg += `<line x1="${padL - 4}" y1="${y}" x2="${padL}" y2="${y}" stroke="#cbd5e1" stroke-width="1"/><text x="${padL - 6}" y="${y + 4}" text-anchor="end" font-size="9" fill="#94a3b8">${val.toFixed(val >= 100 ? 0 : 1)}</text>`;
  }

  const xSkip = Math.max(1, Math.floor(n / 15));
  let xTickSvg = "";
  for (let i = 0; i < n; i += xSkip) {
    xTickSvg += `<text x="${padL + i * gap + gap / 2}" y="${padT + h + 16}" text-anchor="middle" font-size="8" fill="#94a3b8">${labels[i]}</text>`;
  }

  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="font-family:Inter,system-ui,sans-serif">
    <rect x="${padL}" y="${padT}" width="${w}" height="${h}" fill="#fafbfc" rx="4"/>
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + h}" stroke="#e2e8f0" stroke-width="1"/>
    <line x1="${padL}" y1="${padT + h}" x2="${padL + w}" y2="${padT + h}" stroke="#e2e8f0" stroke-width="1"/>
    ${yTickSvg}
    ${xTickSvg}
    ${bars}
    <text x="${padL + w / 2}" y="${height - 2}" text-anchor="middle" font-size="9" fill="#94a3b8">drone</text>
    <text x="10" y="${padT + h / 2}" text-anchor="middle" font-size="9" fill="#94a3b8" transform="rotate(-90,10,${padT + h / 2})">${yLabel}</text>
  </svg>`;
}

export function buildReportText(
  playback: Playback,
  config: SwarmConfig,
  overlays?: PlaybackOverlays,
  report?: SimReport,
): string {
  const r = report ?? computeReport(playback, overlays);
  const gpu = playback.gpuMetrics;
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);
  const lines = [
    "=".repeat(55),
    "            SWARM SIMULATION REPORT",
    "=".repeat(55),
    "",
    `  Generated: ${now}`,
    "",
    "  --- Configuration ---",
    `  Drones:           ${config.n_drones}`,
    `  Duration:         ${config.duration}s`,
    `  Device:           ${config.device.toUpperCase()}`,
    `  Boundary Mode:    ${config.boundary_mode}`,
    `  Physics:          ${config.physics}`,
    `  Integrator:       ${config.integrator}`,
    "",
    "  --- Flight Metrics ---",
    `  Avg Speed:           ${r.avgSpeed.toFixed(2)} m/s`,
    `  Max Speed:           ${r.maxSpeed.toFixed(2)} m/s`,
    `  Min Distance:        ${r.minDistance.toFixed(3)} m`,
    `  Total Collisions:    ${r.totalCollisions}`,
    `  Near Misses:         ${r.nearMisses}`,
    `  Formation Error:     ${r.formationError.toFixed(3)} m`,
    `  Path Efficiency:     ${(r.pathEfficiency * 100).toFixed(1)}%`,
    `  Coverage:            ${r.coveragePercent}%`,
    `  Connectivity:        ${r.connectivityDensity}%`,
    `  Energy Usage:        ${r.energyMetric.toFixed(2)} m²/s²`,
    `  Safety Score:        ${r.safetyScore}%`,
    `  Flight Duration:     ${formatDuration(r.flightDuration)}`,
    `  Total Distance:      ${r.totalDistance.toFixed(1)} m`,
    "",
    `  Collision Timeline: ${r.collisionTimeline.filter((c) => c > 0).length} frames with collisions out of ${r.collisionTimeline.length} total frames`,
    "",
    "  --- Safety & Communication ---",
    `  Area Coverage:       ${r.coveragePercent}%`,
    `  Safety Score:        ${r.safetyScore}% (${r.safetyScore > 80 ? "Good" : r.safetyScore > 50 ? "Fair" : "Poor"})`,
    `  Near Misses:         ${r.nearMisses}`,
    "",
  ];

  if (gpu) {
    lines.push(
      "  --- GPU Benchmark ---",
      `  Platform:            ${gpu.platform.toUpperCase()}  (${gpu.device_name})`,
      `  Wall Time:           ${gpu.sim_time_seconds.toFixed(1)}s`,
      `  Throughput:          ${gpu.timesteps_per_second.toLocaleString()} steps/s`,
      `  Process Memory:      ${gpu.device_memory_mb != null ? gpu.device_memory_mb + " MB" : "—"}`,
      `  Drone Count:         ${gpu.num_drones}`,
      `  Duration:            ${gpu.duration_seconds}s`,
      `  Physics Freq:        ${gpu.physics_freq_hz} Hz`,
      `  Control Freq:        ${gpu.control_freq_hz} Hz`,
      "",
    );
  }

  lines.push(
    "=".repeat(55),
    "  End of Report",
    "=".repeat(55),
  );
  return lines.join("\n");
}

export function buildReportHtml(
  playback: Playback,
  config: SwarmConfig,
  overlays?: PlaybackOverlays,
  report?: SimReport,
): string {
  const r = report ?? computeReport(playback, overlays);
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);
  const maxColl = Math.max(...r.collisionTimeline, 1);
  const timelineBars = r.collisionTimeline.map((c, i) => {
    const h = Math.max(2, (c / maxColl) * 100);
    return `<div style="flex:1;min-width:2px;height:${h}%;background:#2563eb;border-radius:1px 1px 0 0" title="t=${i / playback.timestamps[playback.timestamps.length - 1]}: ${c} collisions"></div>`;
  }).join("");

  const scoreColor = (v: number) => v > 80 ? "#16a34a" : v > 50 ? "#d97706" : "#dc2626";
  const gpu = playback.gpuMetrics;

  const chartW = 800;
  const chartH = 200;

  const speedChart = svgLineChart(r.speedTimeline, chartW, chartH, "#2563eb", "m/s");
  const formationChart = svgLineChart(r.formationErrorTimeline, chartW, chartH, "#7c3aed", "m");
  const collisionChart = svgLineChart(r.collisionTimeline, chartW, chartH, "#dc2626", "collisions");
  const connectivityChart = svgLineChart(r.connectivityTimeline, chartW, chartH, "#16a34a", "%");
  const pathEffChart = svgBarChart(
    r.pathEfficiencyPerDrone.map((v) => v * 100),
    r.pathEfficiencyPerDrone.map((_, i) => `#${i + 1}`),
    chartW,
    chartH,
    "#2563eb",
    "%",
  );
  const nearMissChart = svgLineChart(r.nearMissTimeline, chartW, chartH, "#d97706", "pairs");

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Simulation Report</title>
<style>
  body { font-family: Inter, system-ui, sans-serif; color: #0f172a; background: #fff; padding: 32px; max-width: 900px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.02em; }
  .meta { color: #64748b; font-size: 0.85rem; margin-bottom: 24px; }
  h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; margin: 24px 0 10px; border-bottom: 1px solid #f1f5f9; padding-bottom: 4px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
  td, th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
  th { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
  td { font-size: 0.9rem; }
  .value { font-weight: 700; font-variant-numeric: tabular-nums; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; page-break-inside: avoid; }
  .chart-cell { display: flex; flex-direction: column; }
  .chart-cell svg { max-width: 100%; height: auto; }
  .chart-label { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
  .footer { margin-top: 32px; color: #94a3b8; font-size: 0.75rem; text-align: center; }
  .safety-summary { display: flex; gap: 24px; justify-content: space-around; padding: 16px 0; margin-bottom: 16px; border: 1px solid #f1f5f9; border-radius: 10px; background: #fafbfc; }
  .safety-stat { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .safety-stat-value { font-size: 1.5rem; font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; }
  .safety-stat-label { font-size: 0.68rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
  @media print { body { padding: 16px; } .chart-grid { page-break-inside: avoid; } }
</style>
</head>
<body>
  <h1>Swarm Simulation Report</h1>
  <p class="meta">Generated: ${now} &middot; ${config.n_drones} drones &middot; ${config.duration}s &middot; ${config.device.toUpperCase()}</p>

  <h2>Configuration</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th><th>Parameter</th><th>Value</th></tr>
    <tr><td>Drones</td><td class="value">${config.n_drones}</td><td>Duration</td><td class="value">${config.duration}s</td></tr>
    <tr><td>Device</td><td class="value">${config.device.toUpperCase()}</td><td>Boundary Mode</td><td class="value">${config.boundary_mode}</td></tr>
    <tr><td>Physics</td><td class="value">${config.physics}</td><td>Integrator</td><td class="value">${config.integrator}</td></tr>
  </table>

  <h2>Flight Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
    <tr><td>Average Speed</td><td class="value">${r.avgSpeed.toFixed(2)} m/s</td><td>Maximum Speed</td><td class="value">${r.maxSpeed.toFixed(2)} m/s</td></tr>
    <tr><td>Minimum Distance</td><td class="value">${r.minDistance.toFixed(3)} m</td><td>Total Collisions</td><td class="value" style="color:#dc2626">${r.totalCollisions.toLocaleString()}</td></tr>
    <tr><td>Near Misses</td><td class="value" style="color:#d97706">${r.nearMisses.toLocaleString()}</td><td>Formation Error</td><td class="value">${r.formationError.toFixed(3)} m</td></tr>
    <tr><td>Path Efficiency</td><td class="value">${(r.pathEfficiency * 100).toFixed(1)}%</td><td>Coverage %</td><td class="value">${r.coveragePercent}%</td></tr>
    <tr><td>Connectivity</td><td class="value">${r.connectivityDensity}%</td><td>Energy Usage</td><td class="value">${r.energyMetric.toFixed(2)} m&sup2;/s&sup2;</td></tr>
    <tr><td>Safety Score</td><td class="value" style="color:${scoreColor(r.safetyScore)}">${r.safetyScore}%</td><td>Flight Duration</td><td class="value">${formatDuration(r.flightDuration)}</td></tr>
    <tr><td>Total Distance</td><td class="value" colspan="3">${r.totalDistance.toFixed(1)} m</td></tr>
  </table>

  <h2>Performance Charts</h2>
  <div class="chart-grid">
    <div class="chart-cell">
      <span class="chart-label">Mean Speed over Time</span>
      ${speedChart}
    </div>
    <div class="chart-cell">
      <span class="chart-label">Formation Error over Time</span>
      ${formationChart}
    </div>
    <div class="chart-cell">
      <span class="chart-label">Path Efficiency per Drone</span>
      ${pathEffChart}
    </div>
  </div>

  <h2>Safety &amp; Communication</h2>
  <div class="safety-summary">
    <div class="safety-stat">
      <span class="safety-stat-value">${r.coveragePercent}%</span>
      <span class="safety-stat-label">Area Coverage</span>
    </div>
    <div class="safety-stat">
      <span class="safety-stat-value" style="color:${scoreColor(r.safetyScore)}">${r.safetyScore}%</span>
      <span class="safety-stat-label">Safety Score</span>
    </div>
    <div class="safety-stat">
      <span class="safety-stat-value" style="color:#d97706">${r.nearMisses.toLocaleString()}</span>
      <span class="safety-stat-label">Near Misses</span>
    </div>
  </div>
  <div class="chart-grid">
    <div class="chart-cell">
      <span class="chart-label">Collisions per Frame</span>
      ${collisionChart}
    </div>
    <div class="chart-cell">
      <span class="chart-label">Near Misses per Frame</span>
      ${nearMissChart}
    </div>
    <div class="chart-cell">
      <span class="chart-label">Connectivity Density</span>
      ${connectivityChart}
    </div>
  </div>

  ${gpu ? `
  <h2>GPU Benchmarks</h2>
  <table>
    <tr><th>Metric</th><th>Value</th><th>Metric</th><th>Value</th></tr>
    <tr><td>Compute Platform</td><td class="value">${gpu.platform.toUpperCase()}</td><td>Device Name</td><td class="value">${gpu.device_name}</td></tr>
    <tr><td>Wall Time</td><td class="value">${gpu.sim_time_seconds.toFixed(1)}s</td><td>Throughput</td><td class="value">${gpu.timesteps_per_second.toLocaleString()} steps/s</td></tr>
    <tr><td>Process Memory</td><td class="value">${gpu.device_memory_mb != null ? gpu.device_memory_mb + ' MB' : '—'}</td><td>Drone Count</td><td class="value">${gpu.num_drones}</td></tr>
    <tr><td>Duration</td><td class="value">${gpu.duration_seconds}s</td><td>Physics Freq</td><td class="value">${gpu.physics_freq_hz} Hz</td></tr>
    <tr><td>Control Freq</td><td class="value" colspan="3">${gpu.control_freq_hz} Hz</td></tr>
  </table>
  ` : ""}

  <div class="footer">DroneMD &mdash; Generated by DroneMD</div>
</body>
</html>`;
}

export function downloadReportTxt(
  playback: Playback,
  config: SwarmConfig,
  overlays?: PlaybackOverlays,
  report?: SimReport,
) {
  const text = buildReportText(playback, config, overlays, report);
  download(text, `swarm_report_${config.n_drones}d.txt`, "text/plain");
}

export function downloadReportPdf(
  playback: Playback,
  config: SwarmConfig,
  overlays?: PlaybackOverlays,
  report?: SimReport,
) {
  const html = buildReportHtml(playback, config, overlays, report);
  const win = window.open("", "_blank");
  if (!win) return;
  win.document.write(html);
  win.document.close();
  win.focus();
  win.print();
}

export function downloadCSV(playback: Playback, config: SwarmConfig) {
  const rows = ["time,drone_id,x,y,z"];
  const { timestamps, states, numDrones } = playback;
  for (let t = 0; t < timestamps.length; t++) {
    const time = timestamps[t].toFixed(3);
    for (let d = 0; d < numDrones; d++) {
      const [x, y, z] = states[t][d].slice(0, 3);
      rows.push(`${time},${d},${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)}`);
    }
  }
  download(rows.join("\n"), `swarm_${config.n_drones}d_${config.duration}s.csv`, "text/csv");
}

export function downloadJSONWaypoints(playback: Playback, config: SwarmConfig) {
  const { timestamps, states, numDrones } = playback;
  const waypoints: Record<string, Record<string, [number, number, number]>> = {};
  for (let t = 0; t < timestamps.length; t++) {
    const timeKey = timestamps[t].toFixed(3);
    const frame: Record<string, [number, number, number]> = {};
    for (let d = 0; d < numDrones; d++) {
      const [x, y, z] = states[t][d].slice(0, 3);
      frame[`drone_${d}`] = [x, y, z];
    }
    waypoints[timeKey] = frame;
  }
  const json = JSON.stringify({ config: { n_drones: config.n_drones, duration: config.duration }, waypoints }, null, 2);
  download(json, `swarm_${config.n_drones}d_${config.duration}s.json`, "application/json");
}

export function downloadROS(playback: Playback, config: SwarmConfig) {
  const rows = ["# ROS Crazyflie waypoint file", "# time(sec), drone_id, x(m), y(m), z(m), yaw(rad)"];
  const { timestamps, states, numDrones } = playback;
  for (let t = 0; t < timestamps.length; t++) {
    const time = timestamps[t].toFixed(3);
    for (let d = 0; d < numDrones; d++) {
      const [x, y, z] = states[t][d].slice(0, 3);
      rows.push(`${time},${d},${x.toFixed(4)},${y.toFixed(4)},${z.toFixed(4)},0.0`);
    }
  }
  download(rows.join("\n"), `swarm_${config.n_drones}d_${config.duration}s_ros.csv`, "text/csv");
}
