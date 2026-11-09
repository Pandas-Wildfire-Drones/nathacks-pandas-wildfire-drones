# Skylink Drone Fleet Dashboard

A responsive HTML + TypeScript dashboard for monitoring multiple mostly-autonomous drones. It is intentionally framework-free and uses Vite only for local development and TypeScript compilation.

## Run locally

```bash
npm install
npm run dev
```

Open the local Vite URL printed in the terminal. Without a backend, the dashboard shows clearly illustrative sample telemetry and will try to reconnect to the default WebSocket endpoint.

Build the static site:

```bash
npm run build
```

The compiled site is written to `dist/`.

## Connect your backend

By default, the UI expects:

- WebSocket telemetry: `ws://localhost:8000/ws`
- Return-all API: `POST http://localhost:8000/api/fleet/return`

Override these URLs with query parameters when opening the UI:

```text
http://localhost:5173/?ws=ws://127.0.0.1:9000/ws&api=http://127.0.0.1:9000
```

The WebSocket endpoint should send JSON messages in one of these formats.

### Full snapshot

Send this periodically, or whenever you want to replace the current fleet state:

```json
{
  "type": "snapshot",
  "generated_at": "2026-11-08T23:00:00Z",
  "drones": [
    {
      "id": "drone-01",
      "name": "Scout 01",
      "status": "autonomous",
      "battery_percent": 82,
      "location": {
        "latitude": 53.5461,
        "longitude": -113.4938,
        "altitude_m": 42
      },
      "heading_deg": 128,
      "last_seen": "2026-11-08T22:59:58Z",
      "message": "Scanning assigned area"
    }
  ],
  "images": [
    {
      "id": "capture-1042",
      "drone_id": "drone-01",
      "captured_at": "2026-11-08T22:58:12Z",
      "image_url": "http://localhost:8000/captures/capture-1042.jpg",
      "location": {
        "latitude": 53.5463,
        "longitude": -113.4935,
        "altitude_m": 41
      },
      "label": "Survey image"
    }
  ]
}
```

### Individual updates (optional)

Instead of sending a full snapshot every time, the backend may send individual messages:

```json
{
  "type": "drone_update",
  "drone": {
    "id": "drone-01",
    "name": "Scout 01",
    "status": "returning",
    "battery_percent": 76,
    "location": { "latitude": 53.5462, "longitude": -113.4937, "altitude_m": 35 },
    "heading_deg": 270,
    "last_seen": "2026-11-08T23:01:00Z",
    "message": "Return requested"
  }
}
```

```json
{
  "type": "image_captured",
  "image": {
    "id": "capture-1043",
    "drone_id": "drone-01",
    "captured_at": "2026-11-08T23:01:04Z",
    "image_url": "http://localhost:8000/captures/capture-1043.jpg",
    "location": { "latitude": 53.5464, "longitude": -113.4936, "altitude_m": 39 },
    "label": "New capture"
  }
}
```

`image_url` must be reachable by the user's browser. If the image server is on a different origin, configure appropriate CORS headers.

## Return-all command contract

When the operator confirms **Return all drones**, the UI sends:

```http
POST /api/fleet/return
Content-Type: application/json
```

```json
{
  "command": "return_all",
  "issued_at": "2026-11-08T23:02:00.000Z"
}
```

Your backend should authenticate and authorize the operator, validate fleet state, and independently decide whether/how a return request can be accepted. The UI only reports whether the HTTP endpoint accepted the request; it does not assume the drones have returned. The dashboard should receive updated drone statuses over the WebSocket.

## Notes

- `status` values used by the UI include `autonomous`, `returning`, `idle`, `offline`, and `warning`. Other strings are displayed as supplied.
- `battery_percent` is a number from 0 to 100.
- Coordinates are decimal degrees. `altitude_m` and `heading_deg` are optional.
- `last_seen` and `captured_at` should be ISO 8601 timestamps.
- The sample coordinates and sample drone records are for UI preview only; replace them with live telemetry.
