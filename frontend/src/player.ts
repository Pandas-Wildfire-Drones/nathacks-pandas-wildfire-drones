import * as THREE from "three";

/**
 * Client-side flight model for the player's drone (sent to the server as a pose at 20 Hz).
 * Deliberately crude (the project leaves real navigation to PX4): WASD in the body frame, R/F altitude with hard
 * min/max limits, Q/E or mouse to yaw, Shift to boost. Speeds/accel come from the contracts PlatformSpec.
 */
export interface FlightLimits {
  minZ: number;
  maxZ: number;
  cruise: number;
  max: number;
  accel: number;
  vz: number;
  radius: number;
}

export class Player {
  x = 0;
  y = 0;
  z = 40;
  yaw = 0; // compass degrees
  pitch = -14; // camera look, degrees
  vx = 0;
  vy = 0;
  vz = 0;
  dropping = false;
  enabled = false;
  limits: FlightLimits = { minZ: 30, maxZ: 150, cruise: 15, max: 20, accel: 4, vz: 6, radius: 170 };
  private keys = new Set<string>();
  private mouseDx = 0;
  private mouseDy = 0;

  constructor(private canvas: HTMLCanvasElement) {
    window.addEventListener("keydown", (e) => {
      if (["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "KeyR", "KeyF", "Space", "ShiftLeft", "ShiftRight", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) {
        if (this.enabled) e.preventDefault();
      }
      this.keys.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());
    document.addEventListener("mousemove", (e) => {
      if (document.pointerLockElement === this.canvas) {
        this.mouseDx += e.movementX;
        this.mouseDy += e.movementY;
      }
    });
  }

  lock(): void {
    if (document.pointerLockElement !== this.canvas) this.canvas.requestPointerLock?.();
  }

  spawn(x: number, y: number, z: number, yaw: number, limits: FlightLimits, pitch: number): void {
    this.x = x;
    this.y = y;
    this.z = z;
    this.yaw = yaw;
    this.pitch = pitch;
    this.vx = this.vy = this.vz = 0;
    this.limits = limits;
  }

  get speed(): number {
    return Math.hypot(this.vx, this.vy);
  }

  update(dt: number): void {
    const k = this.keys;
    if (!this.enabled) {
      this.mouseDx = this.mouseDy = 0;
      return;
    }
    const L = this.limits;
    const fwd = (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0) - (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0);
    const str = (k.has("KeyD") ? 1 : 0) - (k.has("KeyA") ? 1 : 0);
    const turn = (k.has("KeyE") || k.has("ArrowRight") ? 1 : 0) - (k.has("KeyQ") || k.has("ArrowLeft") ? 1 : 0);
    const up = (k.has("KeyR") ? 1 : 0) - (k.has("KeyF") ? 1 : 0);
    this.yaw = (this.yaw + turn * 75 * dt + this.mouseDx * 0.12 + 360) % 360;
    this.pitch = Math.max(-80, Math.min(10, this.pitch - this.mouseDy * 0.1));
    this.mouseDx = this.mouseDy = 0;
    const boost = k.has("ShiftLeft") || k.has("ShiftRight");
    const vmax = boost ? L.max : L.cruise;
    const yr = (this.yaw * Math.PI) / 180;
    const fx = Math.sin(yr), fy = Math.cos(yr); // forward (east, north)
    const rx = Math.cos(yr), ry = -Math.sin(yr); // right
    let dx = fwd * fx + str * rx;
    let dy = fwd * fy + str * ry;
    const n = Math.hypot(dx, dy);
    if (n > 1) {
      dx /= n;
      dy /= n;
    }
    const tx = dx * vmax, ty = dy * vmax;
    const ex = tx - this.vx, ey = ty - this.vy;
    const e = Math.hypot(ex, ey);
    const a = L.accel * (n > 0 ? 1 : 1.4) * dt;
    if (e <= a) {
      this.vx = tx;
      this.vy = ty;
    } else {
      this.vx += (ex / e) * a;
      this.vy += (ey / e) * a;
    }
    const tvz = up * L.vz;
    this.vz += Math.max(-L.vz * 2 * dt, Math.min(L.vz * 2 * dt, tvz - this.vz));
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.z = Math.max(L.minZ, Math.min(L.maxZ, this.z + this.vz * dt));
    if (this.z <= L.minZ || this.z >= L.maxZ) this.vz = 0;
    const r = Math.hypot(this.x, this.y);
    if (r > L.radius) {
      this.x *= L.radius / r;
      this.y *= L.radius / r;
      const nx = this.x / r, ny = this.y / r;
      const out = this.vx * nx + this.vy * ny;
      if (out > 0) {
        this.vx -= out * nx;
        this.vy -= out * ny;
      }
    }
    this.dropping = k.has("Space");
  }

  applyCamera(cam: THREE.PerspectiveCamera): void {
    cam.position.set(this.x, this.z, -this.y);
    const yr = (this.yaw * Math.PI) / 180;
    // body-frame velocity -> slight bank/pitch for feel
    const fwdV = this.vx * Math.sin(yr) + this.vy * Math.cos(yr);
    const sideV = this.vx * Math.cos(yr) - this.vy * Math.sin(yr);
    cam.rotation.order = "YXZ";
    cam.rotation.y = -yr;
    cam.rotation.x = (this.pitch * Math.PI) / 180 - fwdV * 0.004;
    cam.rotation.z = -sideV * 0.01;
  }

  pose(): object {
    return { type: "input", x: this.x, y: this.y, z: this.z, yaw: this.yaw, v: this.speed, drop: this.dropping };
  }
}
