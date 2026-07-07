import { Pause, Play, RotateCcw, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { Playback, PlaybackOverlays } from "./types";

type PlayerProps = {
  playback: Playback;
  overlays?: PlaybackOverlays;
  onClose: () => void;
  autoPlay?: boolean;
  embedded?: boolean;
  loop?: boolean;
};

type DroneScene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  drones: THREE.Group[];
  trails: THREE.Line[];
  trailColors: THREE.Color[];
  collisionRings: THREE.Mesh[];
  deckMaterials: THREE.MeshStandardMaterial[];
  defaultEmissive: number[];
  animationId: number | null;
  lastUiUpdate: number;
};

const TRAIL_SECONDS = 2.4;
const TRAIL_SAMPLES = 48;
const STL_SCALE = 0.001;

const COLLISION_GLOW_RADIUS = 0.05;
const COLLISION_GLOW_OPACITY_BASE = 0.25;
const COLLISION_GLOW_OPACITY_AMP = 0.3;
const COLLISION_GLOW_PULSE_HZ = 10;
const COLLISION_EMISSIVE_INTENSITY = 1.5;
const geometryCache = new Map<string, Promise<THREE.BufferGeometry>>();

type MeshPart = {
  file: string;
  color: number;
  position?: [number, number, number];
  rotationZ?: number;
  rotationX?: number;
};

const CF21B_STATIC_PARTS: MeshPart[] = [
  { file: "cf21B/cf21B_pcb.stl", color: 0x4d4d4d },
  { file: "cf21B/cf21B_motors.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_prop-guards.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_connectors.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_connector-pins.stl", color: 0xf7e099 },
  { file: "cf21B/cf21B_battery.stl", color: 0xb3b3b3 },
  { file: "cf21B/cf21B_battery-holder.stl", color: 0x1a1a1a },
  { file: "cf21B/cf21B_PropL.stl", color: 0x85e625, position: [0.03536, -0.03536, 0.012], rotationZ: 45 },
  { file: "cf21B/cf21B_PropR.stl", color: 0x85e625, position: [-0.03536, -0.03536, 0.012], rotationZ: 135 },
  { file: "cf21B/cf21B_PropL.stl", color: 0x85e625, position: [-0.03536, 0.03536, 0.012], rotationZ: 225 },
  { file: "cf21B/cf21B_PropR.stl", color: 0x85e625, position: [0.03536, 0.03536, 0.012], rotationZ: 315 }
];

function findSampleIndex(timestamps: number[], time: number): number {
  if (time <= timestamps[0]) {
    return 0;
  }
  let lo = 0;
  let hi = timestamps.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (timestamps[mid] <= time) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return Math.max(0, Math.min(timestamps.length - 2, hi));
}

function sampleDroneState(
  playback: Playback,
  time: number,
  droneIndex: number,
  pos: THREE.Vector3,
  quat: THREE.Quaternion
): void {
  const timestamps = playback.timestamps;
  if (timestamps.length === 1) {
    const only = playback.states[0][droneIndex];
    pos.set(only[0], only[1], only[2]);
    quat.set(only[3], only[4], only[5], only[6]).normalize();
    return;
  }

  const clampedTime = Math.max(timestamps[0], Math.min(time, timestamps[timestamps.length - 1]));
  const index = findSampleIndex(timestamps, clampedTime);
  const t0 = timestamps[index];
  const t1 = timestamps[index + 1];
  const alpha = t1 > t0 ? (clampedTime - t0) / (t1 - t0) : 0;
  const a = playback.states[index][droneIndex];
  const b = playback.states[index + 1][droneIndex];

  pos.set(
    THREE.MathUtils.lerp(a[0], b[0], alpha),
    THREE.MathUtils.lerp(a[1], b[1], alpha),
    THREE.MathUtils.lerp(a[2], b[2], alpha)
  );

  const qa = new THREE.Quaternion(a[3], a[4], a[5], a[6]).normalize();
  const qb = new THREE.Quaternion(b[3], b[4], b[5], b[6]).normalize();
  quat.copy(qa).slerp(qb, alpha);
}

function loadGeometry(file: string): Promise<THREE.BufferGeometry> {
  if (!geometryCache.has(file)) {
    const loader = new STLLoader();
    geometryCache.set(
      file,
      new Promise((resolve, reject) => {
        loader.load(`/api/assets/drone/${file}`, (geometry) => {
          geometry.computeVertexNormals();
          resolve(geometry);
        }, undefined, reject);
      })
    );
  }
  return geometryCache.get(file)!;
}

function addMeshPart(parent: THREE.Group, part: MeshPart, material: THREE.Material): void {
  const partGroup = new THREE.Group();
  if (part.position) {
    partGroup.position.set(part.position[0], part.position[1], part.position[2]);
  }
  if (part.rotationX) {
    partGroup.rotation.x = THREE.MathUtils.degToRad(part.rotationX);
  }
  if (part.rotationZ) {
    partGroup.rotation.z = THREE.MathUtils.degToRad(part.rotationZ);
  }
  parent.add(partGroup);

  loadGeometry(part.file).then((geometry) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.setScalar(STL_SCALE);
    partGroup.add(mesh);
  }).catch(() => {
    // Keep the player usable if a mesh asset is missing.
  });
}

function makeDrone(color: THREE.Color): { group: THREE.Group; deckMaterial: THREE.MeshStandardMaterial } {
  const group = new THREE.Group();
  for (const part of CF21B_STATIC_PARTS) {
    addMeshPart(
      group,
      part,
      new THREE.MeshStandardMaterial({ color: part.color, roughness: 0.5, metalness: 0.05 })
    );
  }

  const deckMaterial = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.45,
    roughness: 0.35,
    transparent: true,
    opacity: 0.92
  });
  addMeshPart(group, { file: "cf21B/cf_led-diffusor.stl", color: 0xffffff, position: [0, 0, 0.015], rotationX: 180 }, deckMaterial);
  addMeshPart(group, { file: "cf21B/cf_led-diffusor.stl", color: 0xffffff, position: [0, 0, -0.002] }, deckMaterial);
  return { group, deckMaterial };
}

function makeFlightArea(playback: Playback): THREE.Group {
  const group = new THREE.Group();
  const [minX, minY, minZ] = playback.bounds.min;
  const [maxX, maxY] = playback.bounds.max;
  const width = maxX - minX;
  const depth = maxY - minY;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.9,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide
    })
  );
  floor.position.set(centerX, centerY, minZ);
  group.add(floor);

  const lineVertices: number[] = [];
  const addLine = (a: [number, number, number], b: [number, number, number]) => {
    lineVertices.push(...a, ...b);
  };
  addLine([minX, minY, minZ + 0.002], [maxX, minY, minZ + 0.002]);
  addLine([maxX, minY, minZ + 0.002], [maxX, maxY, minZ + 0.002]);
  addLine([maxX, maxY, minZ + 0.002], [minX, maxY, minZ + 0.002]);
  addLine([minX, maxY, minZ + 0.002], [minX, minY, minZ + 0.002]);

  const spacing = 0.5;
  for (let x = Math.ceil(minX / spacing) * spacing; x <= maxX; x += spacing) {
    addLine([x, minY, minZ + 0.001], [x, maxY, minZ + 0.001]);
  }
  for (let y = Math.ceil(minY / spacing) * spacing; y <= maxY; y += spacing) {
    addLine([minX, y, minZ + 0.001], [maxX, y, minZ + 0.001]);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(lineVertices, 3));
  group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.75 })));
  return group;
}

export function Player({ playback, overlays, onClose, autoPlay = false, embedded = false, loop = false }: PlayerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<DroneScene | null>(null);
  const playheadRef = useRef(0);
  const wallClockRef = useRef(0);
  const playingRef = useRef(false);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { playingRef.current = playing; }, [playing]);
  const duration = useMemo(
    () => playback.timestamps[playback.timestamps.length - 1] ?? 0,
    [playback.timestamps]
  );

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe2e8f0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(2.8, -3.2, 2.4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.9);
    controls.enableDamping = true;

    scene.add(new THREE.HemisphereLight(0xdbeee7, 0x23322d, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(2.5, -2, 4);
    scene.add(key);

    scene.add(makeFlightArea(playback));

    const drones: THREE.Group[] = [];
    const trails: THREE.Line[] = [];
    const trailColors: THREE.Color[] = [];
    const collisionRings: THREE.Mesh[] = [];
    const deckMaterials: THREE.MeshStandardMaterial[] = [];
    const defaultEmissive: number[] = [];
    playback.colors.forEach((rgb, index) => {
      const color = new THREE.Color(rgb[0], rgb[1], rgb[2]);
      const { group, deckMaterial } = makeDrone(color);
      drones.push(group);
      deckMaterials.push(deckMaterial);
      defaultEmissive.push(deckMaterial.emissiveIntensity);
      scene.add(group);

      const trailGeometry = new THREE.BufferGeometry();
      trailGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(TRAIL_SAMPLES * 3), 3)
      );
      const trailMat = new THREE.LineBasicMaterial({ color: color.clone(), transparent: true, opacity: 0.65 });
      const trail = new THREE.Line(trailGeometry, trailMat);
      trail.name = `trail-${index}`;
      trails.push(trail);
      trailColors.push(color.clone());
      scene.add(trail);

      const glowGeo = new THREE.SphereGeometry(COLLISION_GLOW_RADIUS, 16, 16);
      const glowMat = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        emissive: 0xff0000,
        emissiveIntensity: 2.0,
        transparent: true,
        opacity: 0,
      });
      const glow = new THREE.Mesh(glowGeo, glowMat);
      glow.visible = false;
      group.add(glow);
      collisionRings.push(glow);
    });

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width;
        const height = entry.contentRect.height;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
      }
    });
    resizeObserver.observe(mount);

    sceneRef.current = {
      renderer,
      scene,
      camera,
      controls,
      drones,
      trails,
      trailColors,
      collisionRings,
      deckMaterials,
      defaultEmissive,
      animationId: null,
      lastUiUpdate: 0
    };

    return () => {
      resizeObserver.disconnect();
      const active = sceneRef.current;
      if (active && active.animationId !== null) {
        cancelAnimationFrame(active.animationId);
      }
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, [playback]);

  useEffect(() => {
    const tempPos = new THREE.Vector3();
    const tempQuat = new THREE.Quaternion();
    const trailPos = new THREE.Vector3();
    const trailQuat = new THREE.Quaternion();

    const renderAt = (time: number) => {
      const active = sceneRef.current;
      if (!active) {
        return;
      }

      let frameIndex = -1;
      if (overlays) {
        frameIndex = findSampleIndex(playback.timestamps, time);
      }
      const collisionSet = new Set<string>();
      if (frameIndex >= 0 && overlays?.collisions_per_frame[frameIndex]) {
        for (const [a, b] of overlays.collisions_per_frame[frameIndex]) {
          collisionSet.add(`${a}`);
          collisionSet.add(`${b}`);
        }
      }
      const allSpeeds = overlays?.speeds ?? null;
      let maxSpeedObserved = 1;
      if (allSpeeds) {
        for (const row of allSpeeds) {
          for (const s of row) {
            if (s > maxSpeedObserved) maxSpeedObserved = s;
          }
        }
      }

      playback.colors.forEach((_, droneIndex) => {
        sampleDroneState(playback, time, droneIndex, tempPos, tempQuat);
        const drone = active.drones[droneIndex];
        drone.position.copy(tempPos);
        drone.quaternion.copy(tempQuat);

        const mat = active.deckMaterials[droneIndex];
        const ring = active.collisionRings[droneIndex];
        const isColliding = collisionSet.has(`${droneIndex}`);

        if (isColliding) {
          mat.emissive.setHex(0xff0000);
          mat.emissiveIntensity = COLLISION_EMISSIVE_INTENSITY;
          ring.visible = true;
          (ring.material as THREE.MeshStandardMaterial).opacity =
            COLLISION_GLOW_OPACITY_BASE + COLLISION_GLOW_OPACITY_AMP * Math.abs(Math.sin(time * COLLISION_GLOW_PULSE_HZ));
        } else {
          ring.visible = false;
          (ring.material as THREE.MeshStandardMaterial).opacity = 0;
          if (allSpeeds && frameIndex >= 0 && frameIndex < allSpeeds.length) {
            const speed = allSpeeds[frameIndex][droneIndex] ?? 0;
            const t = Math.min(1, speed / maxSpeedObserved);
            const c = new THREE.Color();
            if (t < 0.5) {
              c.setHSL(0.6 - t * 1.2, 1, 0.5);
            } else {
              c.setHSL(0.0, 1, 0.5 + (t - 0.5) * 0.5);
            }
            mat.emissive.copy(c);
            mat.emissiveIntensity = 0.35 + t * 0.65;
          } else {
            const rgb = playback.colors[droneIndex];
            mat.emissive.setRGB(rgb[0], rgb[1], rgb[2]);
            mat.emissiveIntensity = active.defaultEmissive[droneIndex];
          }
        }

        const attr = active.trails[droneIndex].geometry.getAttribute("position") as THREE.BufferAttribute;
        for (let i = 0; i < TRAIL_SAMPLES; i += 1) {
          const offset = ((TRAIL_SAMPLES - 1 - i) / (TRAIL_SAMPLES - 1)) * TRAIL_SECONDS;
          sampleDroneState(playback, Math.max(0, time - offset), droneIndex, trailPos, trailQuat);
          attr.setXYZ(i, trailPos.x, trailPos.y, trailPos.z);
        }
        attr.needsUpdate = true;
      });

      active.controls.update();
      active.renderer.render(active.scene, active.camera);
    };

    const animate = () => {
      const active = sceneRef.current;
      if (!active) {
        return;
      }
      let time = playheadRef.current;
      if (playingRef.current) {
        time = (performance.now() - wallClockRef.current) / 1000;
        if (time >= duration) {
          if (loop) {
            wallClockRef.current = performance.now();
            time = 0;
          } else {
            time = duration;
            setPlaying(false);
          }
        }
      }
      playheadRef.current = time;
      renderAt(time);
      const now = performance.now();
      if (now - active.lastUiUpdate > 80) {
        active.lastUiUpdate = now;
        setPlayhead(time);
      }
      active.animationId = requestAnimationFrame(animate);
    };

    const activeScene = sceneRef.current;
    if (activeScene) {
      activeScene.animationId = requestAnimationFrame(animate);
    }
    return () => {
      const active = sceneRef.current;
      if (active && active.animationId !== null) {
        cancelAnimationFrame(active.animationId);
        active.animationId = null;
      }
    };
  }, [duration, playback, loop, overlays]);

  const setTime = (time: number) => {
    const nextTime = Math.max(0, Math.min(time, duration));
    playheadRef.current = nextTime;
    setPlayhead(nextTime);
    if (playingRef.current) {
      wallClockRef.current = performance.now() - nextTime * 1000;
    }
  };

  const togglePlay = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (playheadRef.current >= duration) {
      setTime(0);
    }
    wallClockRef.current = performance.now() - playheadRef.current * 1000;
    setPlaying(true);
  };

  useEffect(() => {
    if (!autoPlay) {
      return;
    }
    wallClockRef.current = performance.now();
    setPlaying(true);
  }, [autoPlay]);

  const restart = () => {
    setTime(0);
    setPlaying(false);
  };

  return (
    <section className={`player-shell${embedded ? " player-embedded" : ""}`}>
      {!embedded && (
        <div className="player-toolbar">
          <div>
            <p className="eyebrow">Browser playback</p>
            <h2>DroneMD</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close player">
            <X size={18} />
          </button>
        </div>
      )}
      <div className="player-canvas" ref={mountRef} />
      <div className="playback-controls">
        <button className="primary-action compact" onClick={togglePlay}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
          {playing ? "Pause" : "Play"}
        </button>
        <button className="secondary-action compact" onClick={restart}>
          <RotateCcw size={18} />
          Restart
        </button>
        <input
          className="timeline"
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={playhead}
          onChange={(event) => setTime(Number(event.target.value))}
        />
        <span className="timecode">
          {playhead.toFixed(1)} / {duration.toFixed(1)}s
        </span>
        {embedded && (
          <button className="icon-button" onClick={onClose} aria-label="Close player">
            <X size={18} />
          </button>
        )}
      </div>
    </section>
  );
}
