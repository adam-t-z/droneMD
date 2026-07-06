import type { Playback, PlaybackOverlays, SimReport, SwarmConfig } from "./types";

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

export function computeReport(playback: Playback, overlays?: PlaybackOverlays): SimReport {
  const { timestamps, states, numDrones } = playback;
  const numFrames = timestamps.length;

  let totalCollisions = 0;
  const collisionTimeline: number[] = [];
  if (overlays) {
    for (const frame of overlays.collisions_per_frame) {
      totalCollisions += frame.length;
      collisionTimeline.push(frame.length);
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
  const safetyScore = numFrames > 0 ? ((numFrames - collisionTimeline.filter((c) => c > 0).length) / numFrames) * 100 : 100;

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

  return {
    totalCollisions,
    avgSpeed,
    maxSpeed,
    minSpeed,
    safetyScore: Math.round(safetyScore),
    flightDuration,
    totalDistance: Math.round(totalDistance * 100) / 100,
    energyMetric: Math.round(energyMetric * 100) / 100,
    collisionTimeline,
  };
}

export function buildReportText(
  playback: Playback,
  config: SwarmConfig,
  overlays?: PlaybackOverlays,
  report?: SimReport,
): string {
  const r = report ?? computeReport(playback, overlays);
  const now = new Date().toISOString().replace("T", " ").slice(0, 19);
  const lines = [
    "=".repeat(50),
    "          SWARM SIMULATION REPORT",
    "=".repeat(50),
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
    "  --- Metrics ---",
    `  Total Collisions: ${r.totalCollisions}`,
    `  Average Speed:    ${r.avgSpeed.toFixed(2)} m/s`,
    `  Maximum Speed:    ${r.maxSpeed.toFixed(2)} m/s`,
    `  Minimum Speed:    ${r.minSpeed.toFixed(2)} m/s`,
    `  Safety Score:     ${r.safetyScore}%`,
    `  Flight Duration:  ${formatDuration(r.flightDuration)}`,
    `  Total Distance:   ${r.totalDistance.toFixed(1)} m`,
    `  Energy Metric:    ${r.energyMetric.toFixed(2)} m^2/s^2`,
    "",
    `  Collision Timeline: ${r.collisionTimeline.filter((c) => c > 0).length} frames with collisions out of ${r.collisionTimeline.length} total frames`,
    "",
    "=".repeat(50),
    "  End of Report",
    "=".repeat(50),
  ];
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

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Swarm Simulation Report</title>
<style>
  body { font-family: Inter, system-ui, sans-serif; color: #0f172a; background: #fff; padding: 32px; max-width: 800px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.02em; }
  .meta { color: #64748b; font-size: 0.85rem; margin-bottom: 24px; }
  h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; margin: 20px 0 10px; }
  table { width: 100%; border-collapse: collapse; }
  td, th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
  th { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
  td { font-size: 0.9rem; }
  .value { font-weight: 700; font-variant-numeric: tabular-nums; }
  .timeline { display: flex; align-items: flex-end; gap: 2px; height: 60px; margin-top: 8px; }
  .footer { margin-top: 32px; color: #94a3b8; font-size: 0.75rem; text-align: center; }
  @media print { body { padding: 16px; } }
</style>
</head>
<body>
  <h1>Swarm Simulation Report</h1>
  <p class="meta">Generated: ${now} &middot; ${config.n_drones} drones &middot; ${config.duration}s duration &middot; ${config.device.toUpperCase()}</p>

  <h2>Configuration</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Drones</td><td class="value">${config.n_drones}</td></tr>
    <tr><td>Duration</td><td class="value">${config.duration}s</td></tr>
    <tr><td>Device</td><td class="value">${config.device.toUpperCase()}</td></tr>
    <tr><td>Boundary Mode</td><td class="value">${config.boundary_mode}</td></tr>
    <tr><td>Physics Model</td><td class="value">${config.physics}</td></tr>
    <tr><td>Integrator</td><td class="value">${config.integrator}</td></tr>
  </table>

  <h2>Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total Collisions</td><td class="value" style="color:#dc2626">${r.totalCollisions.toLocaleString()}</td></tr>
    <tr><td>Average Speed</td><td class="value">${r.avgSpeed.toFixed(2)} m/s</td></tr>
    <tr><td>Maximum Speed</td><td class="value">${r.maxSpeed.toFixed(2)} m/s</td></tr>
    <tr><td>Minimum Speed</td><td class="value">${r.minSpeed.toFixed(2)} m/s</td></tr>
    <tr><td>Safety Score</td><td class="value" style="color:${r.safetyScore > 80 ? "#16a34a" : r.safetyScore > 50 ? "#d97706" : "#dc2626"}">${r.safetyScore}%</td></tr>
    <tr><td>Flight Duration</td><td class="value">${formatDuration(r.flightDuration)}</td></tr>
    <tr><td>Total Distance</td><td class="value">${r.totalDistance.toFixed(1)} m</td></tr>
    <tr><td>Energy Metric</td><td class="value">${r.energyMetric.toFixed(2)} m&sup2;/s&sup2;</td></tr>
  </table>

  ${r.collisionTimeline.length > 0 ? `
  <h2>Collision Timeline</h2>
  <p style="font-size:0.85rem;color:#64748b">${r.collisionTimeline.filter(c => c > 0).length} frames with collisions</p>
  <div class="timeline">${timelineBars}</div>
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
