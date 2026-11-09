export type DroneStatus =
  | "autonomous"
  | "returning"
  | "idle"
  | "offline"
  | "warning"
  | string;

export interface Coordinates {
  latitude: number;
  longitude: number;
  altitude_m?: number;
}

export interface Drone {
  id: string;
  name?: string;
  status: DroneStatus;
  battery_percent: number;
  location: Coordinates;
  heading_deg?: number;
  last_seen: string;
  message?: string;
}

export interface ImageEvent {
  id: string;
  drone_id: string;
  captured_at: string;
  image_url?: string;
  location: Coordinates;
  label?: string;
}

export interface FleetSnapshot {
  type: "snapshot";
  generated_at: string;
  drones: Drone[];
  images: ImageEvent[];
}

export type IncomingMessage =
  | FleetSnapshot
  | { type: "drone_update"; drone: Drone }
  | { type: "image_captured"; image: ImageEvent };
