import "./style.css";
import type { Coordinates, Drone, ImageEvent, IncomingMessage } from "./types";

const WS_URL = new URLSearchParams(window.location.search).get("ws")
  ?? "ws://localhost:8000/ws";
const API_BASE = new URLSearchParams(window.location.search).get("api")
  ?? "http://localhost:8000";
const MAX_IMAGES = 40;

const demoNow = Date.now();
const demoDrones: Drone[] = [
  {
    id: "drone-01", name: "Scout 01", status: "autonomous", battery_percent: 86,
    location: { latitude: 51.0447, longitude: -114.0719, altitude_m: 42 },
    heading_deg: 128, last_seen: new Date(demoNow - 5_000).toISOString(),
    message: "Scanning assigned area",
  },
  {
    id: "drone-02", name: "Scout 02", status: "autonomous", battery_percent: 64,
    location: { latitude: 51.0471, longitude: -114.0672, altitude_m: 38 },
    heading_deg: 244, last_seen: new Date(demoNow - 8_000).toISOString(),
    message: "Following patrol route",
  },
  {
    id: "drone-03", name: "Scout 03", status: "returning", battery_percent: 31,
    location: { latitude: 51.0418, longitude: -114.0753, altitude_m: 25 },
    heading_deg: 12, last_seen: new Date(demoNow - 12_000).toISOString(),
    message: "Return requested · landing site Alpha",
  },
];

const demoImages: ImageEvent[] = [
  {
    id: "capture-1042", drone_id: "drone-01",
    captured_at: new Date(demoNow - 2 * 60_000).toISOString(),
    location: { latitude: 51.0452, longitude: -114.0701, altitude_m: 41 },
    label: "Survey image",
  },
  {
    id: "capture-1041", drone_id: "drone-02",
    captured_at: new Date(demoNow - 7 * 60_000).toISOString(),
    location: { latitude: 51.0483, longitude: -114.0660, altitude_m: 37 },
    label: "Area overview",
  },
  {
    id: "capture-1040", drone_id: "drone-01",
    captured_at: new Date(demoNow - 13 * 60_000).toISOString(),
    location: { latitude: 51.0438, longitude: -114.0738, altitude_m: 43 },
    label: "Survey image",
  },
];

const state = {
  drones: demoDrones,
  images: demoImages,
  connection: "connecting" as "connecting" | "live" | "disconnected",
  generatedAt: new Date(demoNow).toISOString(),
  returningAll: false,
  commandMessage: "",
  commandError: false,
  search: "",
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("App root element #app was not found.");

function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character] ?? character);
}

function validCoordinates(value: unknown): value is Coordinates {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.latitude === "number"
    && Number.isFinite(candidate.latitude)
    && typeof candidate.longitude === "number"
    && Number.isFinite(candidate.longitude);
}

function validDrone(value: unknown): value is Drone {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.id === "string"
    && typeof candidate.status === "string"
    && typeof candidate.battery_percent === "number"
    && Number.isFinite(candidate.battery_percent)
    && validCoordinates(candidate.location)
    && typeof candidate.last_seen === "string";
}

function validImage(value: unknown): value is ImageEvent {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.id === "string"
    && typeof candidate.drone_id === "string"
    && typeof candidate.captured_at === "string"
    && validCoordinates(candidate.location);
}

function formatCoordinate(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(5) : "—";
}

function coordinatesMarkup(location: Coordinates): string {
  return `<span class="coord-line"><span>${formatCoordinate(location.latitude)}</span><span class="coord-separator">,</span><span>${formatCoordinate(location.longitude)}</span></span>`;
}

function timeAgo(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "Time unavailable";
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 10) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function fullTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown time" : date.toLocaleString();
}

function statusClass(status: string): string {
  return ["autonomous", "returning", "idle", "offline", "warning"].includes(status)
    ? status : "unknown";
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    autonomous: "Autonomous", returning: "Returning", idle: "Idle",
    offline: "Offline", warning: "Attention",
  };
  return labels[status] ?? status.replaceAll("_", " ");
}

function batteryClass(level: number): string {
  if (level <= 20) return "critical";
  if (level <= 40) return "low";
  return "healthy";
}

function renderDroneCard(drone: Drone): string {
  const battery = Math.max(0, Math.min(100, drone.battery_percent));
  const status = statusClass(drone.status);
  return `
    <article class="drone-card">
      <div class="drone-card-top">
        <div class="drone-identity">
          <div class="drone-glyph" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
          <div><div class="drone-name">${escapeHtml(drone.name || drone.id)}</div><div class="drone-id">${escapeHtml(drone.id)}</div></div>
        </div>
        <span class="status-pill ${status}"><i></i>${escapeHtml(statusLabel(drone.status))}</span>
      </div>
      <div class="battery-block">
        <div class="battery-label"><span>Battery</span><strong class="${batteryClass(battery)}">${Math.round(battery)}%</strong></div>
        <div class="battery-track"><div class="battery-fill ${batteryClass(battery)}" style="width:${battery}%"></div></div>
      </div>
      <div class="drone-details">
        <div class="detail-cell"><span class="detail-label">DRONE POSITION</span>${coordinatesMarkup(drone.location)}
          <span class="detail-sub">${drone.location.altitude_m != null ? `${escapeHtml(drone.location.altitude_m)} m altitude` : "Altitude unavailable"}</span>
        </div>
        <div class="detail-cell"><span class="detail-label">HEADING</span><strong class="detail-value">${drone.heading_deg != null ? `${Math.round(drone.heading_deg)}°` : "—"}</strong>
          <span class="detail-sub">Updated ${escapeHtml(timeAgo(drone.last_seen))}</span>
        </div>
      </div>
      <div class="drone-card-foot"><span class="live-dot ${status}"></span><span>${escapeHtml(drone.message || "No activity message")}</span></div>
    </article>`;
}

function renderImageCard(image: ImageEvent): string {
  const droneName = state.drones.find((drone) => drone.id === image.drone_id)?.name || image.drone_id;
  const imageContent = image.image_url
    ? `<img src="${escapeHtml(image.image_url)}" alt="${escapeHtml(image.label || "Captured image")}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling?.classList.remove('hidden')" />`
    : "";
  return `
    <article class="image-card">
      <div class="image-preview">
        ${imageContent}
        <div class="image-placeholder ${image.image_url ? "hidden" : ""}" aria-hidden="true"><div class="placeholder-sun"></div><div class="placeholder-hill hill-one"></div><div class="placeholder-hill hill-two"></div><span>IMAGE PREVIEW</span></div>
        <span class="image-type">${escapeHtml(image.label || "Capture")}</span>
      </div>
      <div class="image-card-body">
        <div class="image-title-row"><strong>${escapeHtml(droneName)}</strong><span>${escapeHtml(timeAgo(image.captured_at))}</span></div>
        <div class="image-location-label">CAPTURE COORDINATES</div>
        ${coordinatesMarkup(image.location)}
        <div class="image-card-foot"><span>${escapeHtml(fullTime(image.captured_at))}</span><span class="image-id">${escapeHtml(image.id)}</span></div>
      </div>
    </article>`;
}

function render(): void {
  const online = state.drones.filter((drone) => drone.status !== "offline").length;
  const autonomous = state.drones.filter((drone) => drone.status === "autonomous").length;
  const returning = state.drones.filter((drone) => drone.status === "returning").length;
  const batteries = state.drones.filter((drone) => drone.status !== "offline").map((drone) => drone.battery_percent);
  const averageBattery = batteries.length ? Math.round(batteries.reduce((sum, value) => sum + value, 0) / batteries.length) : 0;
  const query = state.search.trim().toLowerCase();
  const filteredDrones = state.drones.filter((drone) =>
    `${drone.id} ${drone.name ?? ""} ${drone.status}`.toLowerCase().includes(query));
  const sortedImages = [...state.images].sort((a, b) => Date.parse(b.captured_at) - Date.parse(a.captured_at));

  app!.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <a class="brand" href="#" aria-label="Skylink dashboard home">
          <div class="brand-mark"><span></span><span></span><span></span></div>
          <div><strong>SKYLINK</strong><small>FLEET OPERATIONS</small></div>
        </a>
        <div class="topbar-right">
          <div class="connection-status ${state.connection}"><span class="connection-dot"></span><span>${state.connection === "live" ? "LIVE CONNECTION" : state.connection === "connecting" ? "CONNECTING" : "DEMO / OFFLINE"}</span></div>
          <div class="topbar-divider"></div>
          <div class="clock-block"><span class="clock-label">LOCAL TIME</span><strong id="clock">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</strong></div>
        </div>
      </header>

      <main>
        <section class="page-heading">
          <div><div class="eyebrow"><span class="eyebrow-line"></span> MISSION CONTROL</div><h1>Fleet overview<span>.</span></h1><p>Monitor autonomous units, review incoming imagery, and manage fleet-level actions.</p></div>
          <div class="heading-meta"><span class="meta-label">LAST TELEMETRY</span><strong>${escapeHtml(fullTime(state.generatedAt))}</strong><span class="meta-sub">${state.connection === "live" ? "Receiving live JSON telemetry" : "Showing sample data until telemetry connects"}</span></div>
        </section>

        <section class="metrics-grid" aria-label="Fleet summary">
          <div class="metric-card"><div class="metric-top"><span>FLEET SIZE</span><span class="metric-icon">⌘</span></div><div class="metric-value">${state.drones.length.toString().padStart(2, "0")}<small> units</small></div><div class="metric-caption">Registered in this session</div></div>
          <div class="metric-card"><div class="metric-top"><span>ACTIVE UNITS</span><span class="metric-icon green-icon">↗</span></div><div class="metric-value">${online.toString().padStart(2, "0")}<small> / ${state.drones.length.toString().padStart(2, "0")}</small></div><div class="metric-caption"><span class="mini-dot"></span>${autonomous} autonomous · ${returning} returning</div></div>
          <div class="metric-card"><div class="metric-top"><span>AVG. BATTERY</span><span class="metric-icon amber-icon">ϟ</span></div><div class="metric-value">${averageBattery}<small>%</small></div><div class="metric-caption">Across non-offline units</div></div>
          <div class="metric-card"><div class="metric-top"><span>IMAGES RECEIVED</span><span class="metric-icon">▧</span></div><div class="metric-value">${state.images.length.toString().padStart(2, "0")}<small> files</small></div><div class="metric-caption">Latest captures shown below</div></div>
        </section>

        <section class="section-block">
          <div class="section-heading">
            <div><div class="section-kicker">01 / TELEMETRY</div><h2>Drone fleet <span class="count-chip">${state.drones.length}</span></h2></div>
            <div class="fleet-actions">
              <label class="search-box"><span>⌕</span><input id="drone-search" type="search" placeholder="Find a drone…" value="${escapeHtml(state.search)}" aria-label="Search drones" /></label>
              <button class="return-all-button" id="return-all" ${state.returningAll || state.drones.length === 0 ? "disabled" : ""}><span class="return-icon">↶</span>${state.returningAll ? "SENDING COMMAND…" : "RETURN ALL DRONES"}</button>
            </div>
          </div>
          ${state.commandMessage ? `<div class="command-notice ${state.commandError ? "error" : "success"}" role="status">${escapeHtml(state.commandMessage)}</div>` : ""}
          <div class="drone-grid">${filteredDrones.length ? filteredDrones.map(renderDroneCard).join("") : `<div class="empty-state">No drones match “${escapeHtml(state.search)}”.</div>`}</div>
        </section>

        <section class="section-block image-section">
          <div class="section-heading image-heading"><div><div class="section-kicker">02 / VISUAL INTELLIGENCE</div><h2>Recent imagery <span class="count-chip">${state.images.length}</span></h2><p class="section-description">Images are paired with the GPS coordinates recorded at capture time.</p></div><div class="feed-indicator"><span class="mini-dot"></span> INCOMING FEED</div></div>
          <div class="image-grid">${sortedImages.length ? sortedImages.slice(0, 12).map(renderImageCard).join("") : `<div class="empty-state">No images received yet.</div>`}</div>
        </section>
      </main>
      <footer class="footer"><span>SKYLINK OPERATIONS INTERFACE</span><span>TELEMETRY IS PROVIDED BY YOUR BACKEND · ${escapeHtml(state.connection === "live" ? "CONNECTED" : "AWAITING CONNECTION")}</span></footer>
    </div>`;

  const searchInput = document.querySelector<HTMLInputElement>("#drone-search");
  searchInput?.addEventListener("input", (event) => {
    state.search = (event.target as HTMLInputElement).value;
    const cursorPosition = searchInput.selectionStart;
    render();
    const replacement = document.querySelector<HTMLInputElement>("#drone-search");
    replacement?.focus();
    if (cursorPosition !== null) replacement?.setSelectionRange(cursorPosition, cursorPosition);
  });
  document.querySelector<HTMLButtonElement>("#return-all")?.addEventListener("click", sendReturnAll);
  updateClock();
}

async function sendReturnAll(): Promise<void> {
  const available = state.drones.filter((drone) => drone.status !== "offline").length;
  if (!available || state.returningAll) return;
  const confirmed = window.confirm(`Send a return command to the entire fleet?\n\n${available} non-offline drone(s) will receive the command.`);
  if (!confirmed) return;

  state.returningAll = true;
  state.commandMessage = "Sending fleet return command…";
  state.commandError = false;
  render();

  try {
    const response = await fetch(`${API_BASE.replace(/\/$/, "")}/api/fleet/return`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: "return_all", issued_at: new Date().toISOString() }),
    });
    if (!response.ok) throw new Error(`Server responded with HTTP ${response.status}`);
    state.commandMessage = "Return command accepted by the backend. Drone status will update when telemetry arrives.";
    state.commandError = false;
  } catch (error) {
    state.commandMessage = `Could not send return command: ${error instanceof Error ? error.message : "Unknown network error"}. Check the API connection.`;
    state.commandError = true;
  } finally {
    state.returningAll = false;
    render();
  }
}

function applyMessage(message: IncomingMessage | Record<string, unknown>): void {
  if (message.type === "snapshot") {
    const snapshot = message as unknown as Record<string, unknown>;
    if (Array.isArray(snapshot.drones)) {
      state.drones = snapshot.drones.filter(validDrone);
    }
    if (Array.isArray(snapshot.images)) {
      state.images = snapshot.images.filter(validImage).slice(0, MAX_IMAGES);
    }
    if (typeof snapshot.generated_at === "string") state.generatedAt = snapshot.generated_at;
    render();
    return;
  }

  if (message.type === "drone_update") {
    const droneCandidate = (message as Record<string, unknown>).drone;
    if (validDrone(droneCandidate)) {
      const index = state.drones.findIndex((drone) => drone.id === droneCandidate.id);
      if (index >= 0) state.drones[index] = droneCandidate;
      else state.drones.push(droneCandidate);
      state.generatedAt = new Date().toISOString();
      render();
    }
    return;
  }

  if (message.type === "image_captured") {
    const imageCandidate = (message as Record<string, unknown>).image;
    if (validImage(imageCandidate)) {
      state.images = [imageCandidate, ...state.images.filter((image) => image.id !== imageCandidate.id)].slice(0, MAX_IMAGES);
      state.generatedAt = new Date().toISOString();
      render();
    }
  }
}

function connectWebSocket(): void {
  let socket: WebSocket;
  try {
    socket = new WebSocket(WS_URL);
  } catch {
    state.connection = "disconnected";
    render();
    window.setTimeout(connectWebSocket, 4000);
    return;
  }

  socket.addEventListener("open", () => {
    state.connection = "live";
    state.commandMessage = "";
    render();
  });
  socket.addEventListener("message", (event: MessageEvent<string>) => {
    try {
      const parsed: unknown = JSON.parse(event.data);
      if (parsed && typeof parsed === "object") applyMessage(parsed as IncomingMessage | Record<string, unknown>);
    } catch (error) {
      console.error("Ignored invalid JSON telemetry message:", error);
    }
  });
  socket.addEventListener("error", () => socket.close());
  socket.addEventListener("close", () => {
    state.connection = "disconnected";
    render();
    window.setTimeout(connectWebSocket, 4000);
  });
}

function updateClock(): void {
  const clock = document.querySelector<HTMLElement>("#clock");
  if (clock) clock.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

render();
window.setInterval(updateClock, 15_000);
connectWebSocket();
