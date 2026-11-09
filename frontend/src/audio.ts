/**
 * Tiny synthesized sound (WebAudio, no files): night wind, rotor hum that follows speed, phase chime, water rush.
 * Drop a music file at public/audio/ambient.mp3 to add a soundtrack (see exclude/simulator_task.md for sources).
 */
export class Audio {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private hum: { osc: OscillatorNode; gain: GainNode } | null = null;
  private water: GainNode | null = null;
  private music: HTMLAudioElement | null = null;
  muted = false;

  start(): void {
    if (this.ctx) return;
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(ctx.destination);
    const noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const d = noise.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    // wind
    const wind = ctx.createBufferSource();
    wind.buffer = noise;
    wind.loop = true;
    const wf = ctx.createBiquadFilter();
    wf.type = "lowpass";
    wf.frequency.value = 380;
    const wg = ctx.createGain();
    wg.gain.value = 0.05;
    wind.connect(wf).connect(wg).connect(this.master);
    wind.start();
    // rotor hum
    const osc = ctx.createOscillator();
    osc.type = "sawtooth";
    osc.frequency.value = 90;
    const hf = ctx.createBiquadFilter();
    hf.type = "lowpass";
    hf.frequency.value = 420;
    const hg = ctx.createGain();
    hg.gain.value = 0;
    osc.connect(hf).connect(hg).connect(this.master);
    osc.start();
    this.hum = { osc, gain: hg };
    // water rush
    const ws = ctx.createBufferSource();
    ws.buffer = noise;
    ws.loop = true;
    const wbf = ctx.createBiquadFilter();
    wbf.type = "bandpass";
    wbf.frequency.value = 1400;
    wbf.Q.value = 0.6;
    const wsg = ctx.createGain();
    wsg.gain.value = 0;
    ws.connect(wbf).connect(wsg).connect(this.master);
    ws.start();
    this.water = wsg;
    // optional soundtrack
    const m = new window.Audio("/audio/ambient.mp3");
    m.loop = true;
    m.volume = 0.35;
    m.play().then(() => (this.music = m)).catch(() => {});
  }

  setMuted(v: boolean): void {
    this.muted = v;
    if (this.master) this.master.gain.value = v ? 0 : 0.5;
    if (this.music) this.music.muted = v;
  }

  rotor(active: boolean, speed: number, heavy: boolean): void {
    if (!this.ctx || !this.hum) return;
    const t = this.ctx.currentTime;
    this.hum.osc.frequency.setTargetAtTime((heavy ? 55 : 95) + speed * (heavy ? 1.5 : 3), t, 0.2);
    this.hum.gain.gain.setTargetAtTime(active ? (heavy ? 0.06 : 0.035) : 0, t, 0.3);
  }

  waterOn(on: boolean): void {
    if (!this.ctx || !this.water) return;
    this.water.gain.setTargetAtTime(on ? 0.18 : 0, this.ctx.currentTime, 0.08);
  }

  chime(high = false): void {
    if (!this.ctx || !this.master) return;
    const t = this.ctx.currentTime;
    for (const [i, f] of (high ? [660, 880, 1320] : [440, 660]).entries()) {
      const o = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      o.type = "sine";
      o.frequency.value = f;
      g.gain.setValueAtTime(0, t + i * 0.09);
      g.gain.linearRampToValueAtTime(0.12, t + i * 0.09 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, t + i * 0.09 + 0.6);
      o.connect(g).connect(this.master);
      o.start(t + i * 0.09);
      o.stop(t + i * 0.09 + 0.7);
    }
  }
}
