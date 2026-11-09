import type { ServerMsg } from "./types";

/** WebSocket to the sim server (proxied by Vite at /ws). Reconnects on drop. */
export class Net {
  private ws: WebSocket | null = null;
  private queue: string[] = [];
  onMessage: (m: ServerMsg) => void = () => {};
  onStatus: (connected: boolean) => void = () => {};

  connect(): void {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.onStatus(true);
      for (const q of this.queue) ws.send(q);
      this.queue = [];
    };
    ws.onmessage = (ev) => this.onMessage(JSON.parse(ev.data as string) as ServerMsg);
    ws.onclose = () => {
      this.onStatus(false);
      setTimeout(() => this.connect(), 1000);
    };
  }

  send(msg: object): void {
    const s = JSON.stringify(msg);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(s);
    else if ((msg as { type?: string }).type !== "input") this.queue.push(s);
  }
}
