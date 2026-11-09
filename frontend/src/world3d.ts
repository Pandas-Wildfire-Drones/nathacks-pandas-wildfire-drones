import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { DroneRow, HotspotRow, InitMsg } from "./types";

// World (x east, y north, z up)  ->  three (x, y=up, z=south):  (x, z, -y)
export const W = (x: number, y: number, z = 0) => new THREE.Vector3(x, z, -y);

const COL = {
  swarm: new THREE.Color("#4fd1ff"),
  bot: new THREE.Color("#ffb547"),
  player: new THREE.Color("#7dffb2"),
  seeker: new THREE.Color("#b49cff"),
};

function glowTexture(inner = "rgba(255,255,255,1)", outer = "rgba(255,255,255,0)"): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, inner);
  grd.addColorStop(0.35, inner.replace(/,\s*1\)$/, ",0.45)"));
  grd.addColorStop(1, outer);
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const GLOW = glowTexture();

interface DroneVis {
  group: THREE.Group;
  rotors: THREE.Object3D[];
  beacon: THREE.Sprite;
  target: THREE.Vector3;
  yaw: number;
  targetYaw: number;
  kind: "s" | "e";
  isPlayer: boolean;
  seen: number;
}

interface DropFx {
  points: THREE.Points;
  ring: THREE.Mesh;
  t: number;
  life: number;
  fall: number;
  z0: number;
}

interface Marker {
  group: THREE.Group;
  ring: THREE.Mesh;
  beam: THREE.Mesh;
  seen: number;
}

export class World3D {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(70, 1, 0.3, 2500);
  orbit: OrbitControls;
  useOrbit = true;
  init: InitMsg | null = null;
  private root = new THREE.Group();
  private groundTex: THREE.DataTexture | null = null;
  private fireTex: THREE.DataTexture | null = null;
  private groundBase: Uint8Array | null = null;
  private burnt: Uint8Array | null = null;
  private fireMesh: THREE.InstancedMesh | null = null;
  private fireCells: { x: number; y: number; h: number; ph: number }[] = [];
  private fireLights: THREE.PointLight[] = [];
  private drones = new Map<string, DroneVis>();
  private drops: DropFx[] = [];
  private markers = new Map<string, Marker>();
  private lastDropSeq = 0;
  private dummy = new THREE.Object3D();
  hidePlayerMesh = false;
  followId: string | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene.background = new THREE.Color("#060a14");
    this.scene.fog = new THREE.FogExp2("#0b1426", 0.0026);
    this.scene.add(this.root);
    this.orbit = new OrbitControls(this.camera, canvas);
    this.orbit.enableDamping = true;
    this.orbit.maxPolarAngle = Math.PI * 0.47;
    this.orbit.minDistance = 20;
    this.orbit.maxDistance = 700;
    this.camera.position.set(0, 260, 300);
    this.orbit.target.set(0, 0, 0);
    this.addSky();
    window.addEventListener("resize", () => this.resize());
    this.resize();
  }

  resize(): void {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private addSky(): void {
    const hemi = new THREE.HemisphereLight("#5a72a8", "#141a22", 1.5);
    this.scene.add(hemi);
    const moon = new THREE.DirectionalLight("#b9c8ff", 1.0);
    moon.position.set(-300, 400, -200);
    this.scene.add(moon);
    // stars
    const n = 1600;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1;
      const th = Math.random() * Math.PI * 2;
      const r = 1800;
      const s = Math.sqrt(1 - u * u);
      pos[i * 3] = r * s * Math.cos(th);
      pos[i * 3 + 1] = Math.abs(r * u) + 60;
      pos[i * 3 + 2] = r * s * Math.sin(th);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const stars = new THREE.Points(g, new THREE.PointsMaterial({ color: "#cfd8ff", size: 2.2, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.8 }));
    this.scene.add(stars);
    const moonSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(225,232,255,1)"), fog: false, depthWrite: false }));
    moonSprite.position.set(-900, 700, -1100);
    moonSprite.scale.setScalar(160);
    this.scene.add(moonSprite);
  }

  // ---------------------------------------------------------------------------------------------------------------
  build(init: InitMsg): void {
    this.init = init;
    this.scene.remove(this.root);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.drones.clear();
    this.markers.clear();
    this.drops = [];
    this.fireCells = [];
    this.lastDropSeq = 0;
    const R = init.arena_radius;

    // outer ground (outside the geofence)
    const outer = new THREE.Mesh(new THREE.CircleGeometry(R + 420, 64), new THREE.MeshStandardMaterial({ color: "#18211b", roughness: 1 }));
    outer.rotation.x = -Math.PI / 2;
    outer.position.y = -0.08;
    this.root.add(outer);

    // arena ground: colour from the REAL thermal background (FLAME 3), burnt overlay, fire glow as emissive map
    const n = init.ground.n;
    const pre = Uint8Array.from(atob(init.ground.data), (c) => c.charCodeAt(0));
    this.groundBase = pre;
    this.burnt = new Uint8Array(n * n);
    const gdata = new Uint8Array(n * n * 4);
    this.groundTex = new THREE.DataTexture(gdata, n, n, THREE.RGBAFormat);
    this.groundTex.colorSpace = THREE.SRGBColorSpace;
    this.groundTex.magFilter = THREE.LinearFilter;
    this.groundTex.minFilter = THREE.LinearFilter;
    this.fireTex = new THREE.DataTexture(new Uint8Array(n * n * 4), n, n, THREE.RGBAFormat);
    this.fireTex.colorSpace = THREE.SRGBColorSpace;
    this.fireTex.magFilter = THREE.LinearFilter;
    this.fireTex.minFilter = THREE.LinearFilter;
    this.paintGround();
    const size = n * init.ground.cell;
    const groundMat = new THREE.MeshStandardMaterial({
      map: this.groundTex, emissiveMap: this.fireTex, emissive: new THREE.Color("#ffffff"), emissiveIntensity: 2.6, roughness: 0.95,
    });
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(size, size), groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(-R + size / 2, 0, R - size / 2);
    this.root.add(ground);

    // geofence: thin glowing ring + faint wall
    const ring = new THREE.Mesh(new THREE.TorusGeometry(R, 0.35, 6, 160), new THREE.MeshBasicMaterial({ color: "#4fd1ff", transparent: true, opacity: 0.55 }));
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.4;
    this.root.add(ring);
    const wall = new THREE.Mesh(
      new THREE.CylinderGeometry(R, R, 36, 128, 1, true),
      new THREE.MeshBasicMaterial({ color: "#4fd1ff", transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }),
    );
    wall.position.y = 18;
    this.root.add(wall);

    this.buildTrees(init);
    this.buildBase(init);
    this.buildDecoys(init);

    // flames
    const flameGeo = new THREE.ConeGeometry(0.55, 1, 5, 1, true);
    flameGeo.translate(0, 0.5, 0);
    const flameMat = new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
    this.fireMesh = new THREE.InstancedMesh(flameGeo, flameMat, 3000);
    this.fireMesh.count = 0;
    this.fireMesh.frustumCulled = false;
    this.fireMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.root.add(this.fireMesh);
    this.fireLights = [];
    for (let i = 0; i < 6; i++) {
      const l = new THREE.PointLight("#ff7a2e", 0, 55, 1.6);
      l.position.set(0, -50, 0);
      this.root.add(l);
      this.fireLights.push(l);
    }
    this.orbit.target.set(0, 0, 0);
  }

  private buildTrees(init: InitMsg): void {
    const con = init.trees.filter((t) => t.at(-1) === 0);
    const brd = init.trees.filter((t) => t.at(-1) === 1);
    const coneGeo = new THREE.ConeGeometry(1, 1, 6);
    coneGeo.translate(0, 0.5, 0);
    const blobGeo = new THREE.IcosahedronGeometry(1, 0);
    const trunkGeo = new THREE.CylinderGeometry(0.12, 0.18, 1, 5);
    trunkGeo.translate(0, 0.5, 0);
    const leafMat = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 0.9, flatShading: true });
    const trunkMat = new THREE.MeshStandardMaterial({ color: "#2b2118", roughness: 1 });
    const cones = new THREE.InstancedMesh(coneGeo, leafMat, con.length * 2);
    const blobs = new THREE.InstancedMesh(blobGeo, leafMat, Math.max(brd.length, 1));
    const trunks = new THREE.InstancedMesh(trunkGeo, trunkMat, init.trees.length);
    const d = this.dummy;
    const c = new THREE.Color();
    let k = 0;
    con.forEach(([x, y, h, r], i) => {
      for (let tier = 0; tier < 2; tier++) {
        const hh = h * (tier === 0 ? 0.7 : 0.5);
        d.position.copy(W(x, y, h * (tier === 0 ? 0.18 : 0.48)));
        d.scale.set(r * (tier === 0 ? 1 : 0.7), hh, r * (tier === 0 ? 1 : 0.7));
        d.rotation.set(0, (i * 1.7 + tier) % 6, 0);
        d.updateMatrix();
        cones.setMatrixAt(k, d.matrix);
        c.setHSL(0.36 + ((i * 37) % 10) / 300, 0.35, 0.13 + ((i * 13) % 7) / 120);
        cones.setColorAt(k, c);
        k++;
      }
    });
    brd.forEach(([x, y, h, r], i) => {
      d.position.copy(W(x, y, h * 0.62));
      d.scale.set(r * 1.4, h * 0.42, r * 1.4);
      d.rotation.set(i, i * 2, 0);
      d.updateMatrix();
      blobs.setMatrixAt(i, d.matrix);
      c.setHSL(0.27 + ((i * 17) % 10) / 200, 0.3, 0.15);
      blobs.setColorAt(i, c);
    });
    init.trees.forEach(([x, y, h], i) => {
      d.position.copy(W(x, y, 0));
      d.scale.set(1, h * 0.35, 1);
      d.rotation.set(0, 0, 0);
      d.updateMatrix();
      trunks.setMatrixAt(i, d.matrix);
    });
    this.root.add(cones, blobs, trunks);
  }

  private buildBase(init: InitMsg): void {
    const [bx, by] = init.base;
    const pad = new THREE.Mesh(new THREE.CylinderGeometry(22, 22, 0.3, 6), new THREE.MeshStandardMaterial({ color: "#1a2230", roughness: 0.6, metalness: 0.2 }));
    pad.position.copy(W(bx, by + 8, 0.15));
    this.root.add(pad);
    const edge = new THREE.Mesh(new THREE.TorusGeometry(22, 0.25, 4, 6), new THREE.MeshBasicMaterial({ color: "#4fd1ff" }));
    edge.rotation.x = Math.PI / 2;
    edge.rotation.z = Math.PI / 6;
    edge.position.copy(W(bx, by + 8, 0.35));
    this.root.add(edge);
    for (const [px, py] of init.pads) {
      const m = new THREE.Mesh(new THREE.RingGeometry(2.2, 2.6, 24), new THREE.MeshBasicMaterial({ color: "#ffd166", side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2;
      m.position.copy(W(px, py, 0.4));
      this.root.add(m);
    }
    const [sx, sy] = init.station;
    const tank = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 4, 16), new THREE.MeshStandardMaterial({ color: "#1f4e79", emissive: "#0d3b66", emissiveIntensity: 0.6 }));
    tank.position.copy(W(sx, sy, 2));
    this.root.add(tank);
    const tl = new THREE.PointLight("#59b8ff", 30, 40, 1.5);
    tl.position.copy(W(sx, sy, 6));
    this.root.add(tl);
    const bl = new THREE.PointLight("#9fdcff", 40, 70, 1.5);
    bl.position.copy(W(bx, by + 8, 10));
    this.root.add(bl);
  }

  private buildDecoys(init: InitMsg): void {
    for (const [kind, x, y, yaw, area] of init.decoys) {
      const g = new THREE.Group();
      g.position.copy(W(x, y, 0));
      g.rotation.y = (-yaw * Math.PI) / 180;
      if (kind === "warm_rock") {
        const r = Math.sqrt(area / Math.PI);
        const m = new THREE.Mesh(new THREE.DodecahedronGeometry(r, 0), new THREE.MeshStandardMaterial({ color: "#4a4743", flatShading: true, roughness: 1 }));
        m.scale.set(1, 0.45, 0.8);
        m.position.y = r * 0.3;
        g.add(m);
      } else if (kind === "vehicle") {
        const body = new THREE.Mesh(new THREE.BoxGeometry(2, 1.5, 4.5), new THREE.MeshStandardMaterial({ color: "#3a4048", metalness: 0.4, roughness: 0.5 }));
        body.position.y = 0.9;
        g.add(body);
        const cab = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.9, 2), new THREE.MeshStandardMaterial({ color: "#20252b" }));
        cab.position.set(0, 2, -0.4);
        g.add(cab);
        const tail = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.15, 0.05), new THREE.MeshBasicMaterial({ color: "#ff3b3b" }));
        tail.position.set(0, 1.1, 2.27);
        g.add(tail);
      } else {
        const s = Math.sqrt(area);
        const body = new THREE.Mesh(new THREE.BoxGeometry(s, 3, s), new THREE.MeshStandardMaterial({ color: "#3b2f25", roughness: 1 }));
        body.position.y = 1.5;
        g.add(body);
        const roof = new THREE.Mesh(new THREE.ConeGeometry(s * 0.78, 2.2, 4), new THREE.MeshStandardMaterial({ color: "#24201c", flatShading: true }));
        roof.position.y = 4.1;
        roof.rotation.y = Math.PI / 4;
        g.add(roof);
        const win = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 0.8), new THREE.MeshBasicMaterial({ color: "#ffcf73" }));
        win.position.set(0, 1.7, s / 2 + 0.02);
        g.add(win);
        const chim = new THREE.Mesh(new THREE.BoxGeometry(0.7, 2, 0.7), new THREE.MeshStandardMaterial({ color: "#30302e" }));
        chim.position.set(s * 0.25, 5, -s * 0.15);
        g.add(chim);
        const wl = new THREE.PointLight("#ffbf66", 8, 18, 1.8);
        wl.position.set(0, 2, s / 2 + 2);
        g.add(wl);
      }
      this.root.add(g);
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  private paintGround(): void {
    if (!this.groundTex || !this.groundBase || !this.burnt || !this.init) return;
    const n = this.init.ground.n;
    const data = this.groundTex.image.data as Uint8Array;
    const R = this.init.arena_radius;
    const cell = this.init.ground.cell;
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const i = r * n + c;
        const v = this.groundBase[i] / 255;
        const x = -R + (c + 0.5) * cell;
        const y = -R + (r + 0.5) * cell;
        const inside = Math.hypot(x, y) <= R ? 1 : 0.55;
        const b = this.burnt[i] / 255;
        // dark forest floor tinted by the real thermal background; burnt -> char
        let R0 = 40 + 22 * v, G0 = 54 + 26 * v, B0 = 40 + 12 * v;
        R0 = R0 * (1 - b) + 14 * b;
        G0 = G0 * (1 - b) + 11 * b;
        B0 = B0 * (1 - b) + 10 * b;
        data[i * 4] = R0 * inside;
        data[i * 4 + 1] = G0 * inside;
        data[i * 4 + 2] = B0 * inside;
        data[i * 4 + 3] = 255;
      }
    }
    this.groundTex.needsUpdate = true;
  }

  setBurnt(bytes: Uint8Array): void {
    this.burnt = bytes;
    this.paintGround();
  }

  setFire(bytes: Uint8Array, n: number): void {
    if (!this.fireTex || !this.init || !this.fireMesh) return;
    const data = this.fireTex.image.data as Uint8Array;
    const R = this.init.arena_radius;
    const cell = this.init.raster_cell;
    this.fireCells = [];
    for (let i = 0; i < n * n; i++) {
      const h = bytes[i] / 255;
      data[i * 4] = Math.min(255, h * 255);
      data[i * 4 + 1] = Math.min(255, h * h * 120);
      data[i * 4 + 2] = Math.min(255, h * h * h * 30);
      data[i * 4 + 3] = 255;
      if (h > 0.04 && this.fireCells.length < 3000) {
        const r = Math.floor(i / n);
        const c = i % n;
        const hash = Math.sin(i * 12.9898) * 43758.5453;
        const fr = hash - Math.floor(hash);
        this.fireCells.push({ x: -R + (c + 0.2 + 0.6 * fr) * cell, y: -R + (r + 0.8 - 0.6 * fr) * cell, h, ph: fr * 6.28 });
      }
    }
    this.fireTex.needsUpdate = true;
    this.fireMesh.count = this.fireCells.length;
    const col = new THREE.Color();
    this.fireCells.forEach((f, k) => {
      col.setHSL(0.02 + 0.09 * f.h, 1, 0.45 + 0.1 * f.h);
      this.fireMesh!.setColorAt(k, col);
    });
    if (this.fireMesh.instanceColor) this.fireMesh.instanceColor.needsUpdate = true;
    // glow lights at the strongest, well-separated cells
    const sorted = [...this.fireCells].sort((a, b) => b.h - a.h);
    const picks: typeof sorted = [];
    for (const f of sorted) {
      if (picks.length >= this.fireLights.length) break;
      if (picks.every((p) => Math.hypot(p.x - f.x, p.y - f.y) > 22)) picks.push(f);
    }
    this.fireLights.forEach((l, i) => {
      const p = picks[i];
      if (p) {
        l.position.copy(W(p.x, p.y, 3));
        l.intensity = 120 * p.h;
      } else l.intensity = 0;
    });
  }

  setHotspots(rows: HotspotRow[], now: number): void {
    const seen = new Set<string>();
    for (const [id, x, y, area, , status] of rows) {
      seen.add(id);
      let m = this.markers.get(id);
      if (!m) {
        const g = new THREE.Group();
        const ring = new THREE.Mesh(new THREE.RingGeometry(1, 1.25, 40), new THREE.MeshBasicMaterial({ color: "#ffd166", transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.6;
        const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 40, 6, 1, true), new THREE.MeshBasicMaterial({ color: "#ffd166", transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false }));
        beam.position.y = 20;
        g.add(ring, beam);
        this.root.add(g);
        m = { group: g, ring, beam, seen: now };
        this.markers.set(id, m);
      }
      m.group.position.copy(W(x, y, 0));
      const rad = Math.max(2.2, Math.sqrt(Math.max(area, 0.5) / Math.PI) + 1.6);
      m.ring.scale.setScalar(rad);
      const color = status === 3 ? "#5cff9d" : status === 0 ? "#ffd166" : status === 4 ? "#888888" : "#ff5a36";
      (m.ring.material as THREE.MeshBasicMaterial).color.set(color);
      (m.beam.material as THREE.MeshBasicMaterial).color.set(color);
      m.beam.visible = status !== 3;
      m.seen = now;
    }
    for (const [id, m] of this.markers) {
      if (!seen.has(id)) {
        this.root.remove(m.group);
        this.markers.delete(id);
      }
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  private makeDrone(kind: "s" | "e", color: THREE.Color): DroneVis {
    const g = new THREE.Group();
    const s = kind === "e" ? 2.6 : 0.9;
    const bodyMat = new THREE.MeshStandardMaterial({ color: "#2a313c", metalness: 0.5, roughness: 0.45 });
    const accent = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.6 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.1 * s, 0.45 * s, 1.1 * s), bodyMat);
    g.add(body);
    if (kind === "e") {
      const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.55 * s, 0.55 * s, 0.7 * s, 10), new THREE.MeshStandardMaterial({ color: "#1d4f7a", metalness: 0.3, roughness: 0.4 }));
      tank.position.y = -0.55 * s;
      g.add(tank);
    }
    const rotors: THREE.Object3D[] = [];
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + (i * Math.PI) / 2;
      const arm = new THREE.Mesh(new THREE.BoxGeometry(1.5 * s, 0.08 * s, 0.12 * s), bodyMat);
      arm.position.set(Math.cos(a) * 0.75 * s, 0.05 * s, Math.sin(a) * 0.75 * s);
      arm.rotation.y = -a;
      g.add(arm);
      const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.6 * s, 0.6 * s, 0.02 * s, 14), new THREE.MeshBasicMaterial({ color: "#9aa7b8", transparent: true, opacity: 0.22, depthWrite: false }));
      rotor.position.set(Math.cos(a) * 1.45 * s, 0.15 * s, Math.sin(a) * 1.45 * s);
      g.add(rotor);
      rotors.push(rotor);
      const nav = new THREE.Mesh(new THREE.SphereGeometry(0.09 * s, 6, 6), i < 2 ? new THREE.MeshBasicMaterial({ color: i === 0 ? "#ff3b3b" : "#3bff7a" }) : accent);
      nav.position.set(Math.cos(a) * 1.45 * s, -0.05 * s, Math.sin(a) * 1.45 * s);
      g.add(nav);
    }
    const stripe = new THREE.Mesh(new THREE.BoxGeometry(1.14 * s, 0.08 * s, 0.25 * s), accent);
    stripe.position.set(0, 0.24 * s, -0.3 * s);
    g.add(stripe);
    const beacon = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    beacon.scale.setScalar(kind === "e" ? 7 : 4);
    g.add(beacon);
    this.root.add(g);
    return { group: g, rotors, beacon, target: new THREE.Vector3(), yaw: 0, targetYaw: 0, kind, isPlayer: false, seen: 0 };
  }

  setDrones(rows: DroneRow[], mode: "play" | "watch", now: number): void {
    for (const [id, kind, x, y, z, yaw, , , isPlayer] of rows) {
      let d = this.drones.get(id);
      if (!d) {
        const color = isPlayer ? COL.player : kind === "s" ? COL.seeker : mode === "watch" ? COL.swarm : COL.bot;
        d = this.makeDrone(kind, color);
        d.group.position.copy(W(x, y, z));
        d.yaw = yaw;
        this.drones.set(id, d);
      }
      d.isPlayer = isPlayer;
      d.target.copy(W(x, y, z));
      d.targetYaw = yaw;
      d.seen = now;
    }
  }

  dronePosition(id: string): THREE.Vector3 | null {
    return this.drones.get(id)?.group.position ?? null;
  }

  droneIds(): string[] {
    return [...this.drones.keys()];
  }

  addDrops(drops: [number, number, number, number, number][], dropperZ: (x: number, y: number) => number): void {
    for (const [seq, x, y, r, litres] of drops) {
      if (seq <= this.lastDropSeq) continue;
      this.lastDropSeq = seq;
      const z0 = dropperZ(x, y);
      const count = Math.min(140, 20 + Math.round(litres * 0.8));
      const pos = new Float32Array(count * 3);
      const off = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        const a = Math.random() * Math.PI * 2;
        const d = Math.random() * r * 0.6;
        off[i * 3] = Math.cos(a) * d;
        off[i * 3 + 1] = Math.random();
        off[i * 3 + 2] = Math.sin(a) * d;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.userData.off = off;
      g.userData.center = W(x, y, 0);
      const pts = new THREE.Points(g, new THREE.PointsMaterial({ color: "#8fd3ff", size: 0.35, transparent: true, opacity: 0.85, depthWrite: false }));
      pts.frustumCulled = false;
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 48), new THREE.MeshBasicMaterial({ color: "#7cc8ff", transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(W(x, y, 0.5));
      ring.userData.r = r;
      this.root.add(pts, ring);
      this.drops.push({ points: pts, ring, t: 0, life: 2.2, fall: Math.sqrt((2 * Math.max(z0, 2)) / 9.81), z0 });
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  update(dt: number, time: number): void {
    const k = 1 - Math.exp(-dt * 10);
    for (const [id, d] of this.drones) {
      d.group.position.lerp(d.target, k);
      let dy = d.targetYaw - d.yaw;
      dy = ((dy + 540) % 360) - 180;
      d.yaw += dy * k;
      d.group.rotation.y = (-d.yaw * Math.PI) / 180;
      for (const r of d.rotors) r.rotation.y += dt * 40;
      d.group.visible = !(d.isPlayer && this.hidePlayerMesh);
      (d.beacon.material as THREE.SpriteMaterial).opacity = 0.55 + 0.45 * Math.max(0, Math.sin(time * 3 + id.length));
    }
    // flames flicker
    if (this.fireMesh) {
      const dm = this.dummy;
      this.fireCells.forEach((f, i) => {
        const fl = 0.75 + 0.25 * Math.sin(time * 9 + f.ph) + 0.1 * Math.sin(time * 23 + f.ph * 3);
        dm.position.copy(W(f.x, f.y, 0));
        const s = 0.6 + 1.1 * f.h;
        dm.scale.set(s, (0.8 + 2.6 * f.h) * fl, s);
        dm.rotation.set(0, f.ph, 0);
        dm.updateMatrix();
        this.fireMesh!.setMatrixAt(i, dm.matrix);
      });
      this.fireMesh.instanceMatrix.needsUpdate = true;
      for (const l of this.fireLights) if (l.intensity > 0) l.intensity *= 0.97 + 0.06 * Math.random();
    }
    // water drops
    for (let i = this.drops.length - 1; i >= 0; i--) {
      const fx = this.drops[i];
      fx.t += dt;
      const pos = fx.points.geometry.getAttribute("position") as THREE.BufferAttribute;
      const off = fx.points.geometry.userData.off as Float32Array;
      const c = fx.points.geometry.userData.center as THREE.Vector3;
      for (let j = 0; j < pos.count; j++) {
        const tt = Math.max(0, fx.t - off[j * 3 + 1] * 0.4);
        const zz = Math.max(0, fx.z0 - 0.5 * 9.81 * tt * tt);
        const spread = Math.min(1, tt / fx.fall);
        pos.setXYZ(j, c.x + off[j * 3] * spread, zz, c.z + off[j * 3 + 2] * spread);
      }
      pos.needsUpdate = true;
      const landed = Math.max(0, fx.t - fx.fall);
      const r = fx.ring.userData.r as number;
      fx.ring.scale.setScalar(Math.max(0.01, r * Math.min(1.4, 0.3 + landed * 1.4)));
      (fx.ring.material as THREE.MeshBasicMaterial).opacity = landed > 0 ? Math.max(0, 0.8 - landed * 0.55) : 0;
      (fx.points.material as THREE.PointsMaterial).opacity = Math.max(0, 0.85 - Math.max(0, fx.t - fx.fall - 0.3));
      if (fx.t > fx.fall + fx.life) {
        this.root.remove(fx.points, fx.ring);
        fx.points.geometry.dispose();
        this.drops.splice(i, 1);
      }
    }
    // pulse hotspot markers
    for (const m of this.markers.values()) {
      const mat = m.ring.material as THREE.MeshBasicMaterial;
      mat.opacity = 0.55 + 0.35 * Math.sin(time * 4);
    }
    if (this.useOrbit) {
      if (this.followId) {
        const p = this.dronePosition(this.followId);
        if (p) {
          const delta = p.clone().sub(this.orbit.target).multiplyScalar(k);
          this.orbit.target.add(delta);
          this.camera.position.add(delta);
        }
      }
      this.orbit.update();
    }
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }
}
