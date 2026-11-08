import React, { useRef, useEffect } from 'react';
import { Drone, FireSpot, Obstacle, BaseStation, Wind, Vector2D } from '../simulator/types';

interface TacticalRadarProps {
  drones: Drone[];
  fires: FireSpot[];
  obstacles: Obstacle[];
  baseStation: BaseStation;
  wind: Wind;
  selectedDroneId: string | null;
  onSelectDrone: (id: string | null) => void;
  onCanvasClick: (point: Vector2D) => void;
  toolMode: 'inspect' | 'add_fire' | 'add_obstacle' | 'water_drop';
  showCommMesh: boolean;
  showSensorRanges: boolean;
  showTrails: boolean;
}

export const TacticalRadar: React.FC<TacticalRadarProps> = ({
  drones,
  fires,
  obstacles,
  baseStation,
  wind,
  selectedDroneId,
  onSelectDrone,
  onCanvasClick,
  toolMode,
  showCommMesh,
  showSensorRanges,
  showTrails,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const pulseRef = useRef<number>(0);
  const transformRef = useRef<{ scale: number; offsetX: number; offsetY: number }>({
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  });

  // Particles for smoke and water
  const particlesRef = useRef<Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
    maxLife: number;
    type: 'smoke' | 'water' | 'spark';
    color: string;
    size: number;
  }>>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let isMounted = true;

    const WORLD_W = 960;
    const WORLD_H = 640;

    const render = () => {
      if (!isMounted) return;
      pulseRef.current = (pulseRef.current + 0.04) % (Math.PI * 2);
      const pulse = Math.sin(pulseRef.current);

      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      const targetW = Math.max(300, Math.floor(rect.width * dpr));
      const targetH = Math.max(200, Math.floor(rect.height * dpr));

      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      const scale = Math.min(canvas.width / WORLD_W, canvas.height / WORLD_H);
      const offsetX = (canvas.width - WORLD_W * scale) / 2;
      const offsetY = (canvas.height - WORLD_H * scale) / 2;

      transformRef.current = { scale, offsetX, offsetY };

      // 1. Clear background
      ctx.fillStyle = '#060911';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scale, scale);

      // Radar boundary box
      ctx.fillStyle = '#080d19';
      ctx.fillRect(0, 0, WORLD_W, WORLD_H);
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(0, 0, WORLD_W, WORLD_H);

      const width = WORLD_W;
      const height = WORLD_H;

      // 2. Draw tactical radar grid
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Range rings centered at base station
      ctx.strokeStyle = '#1e293b';
      ctx.setLineDash([4, 6]);
      [150, 300, 450, 600].forEach(radius => {
        ctx.beginPath();
        ctx.arc(baseStation.x, baseStation.y, radius, 0, Math.PI * 2);
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // 3. Draw Base Station / Helipad
      const basePulse = Math.sin(pulseRef.current * 1.5) * 3;
      // Outer safety boundary
      ctx.fillStyle = 'rgba(14, 165, 233, 0.06)';
      ctx.beginPath();
      ctx.arc(baseStation.x, baseStation.y, baseStation.radius + basePulse, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(baseStation.x, baseStation.y, baseStation.radius, 0, Math.PI * 2);
      ctx.stroke();

      // Helipad landing circle & H
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(baseStation.x, baseStation.y, 22, 0, Math.PI * 2);
      ctx.stroke();

      ctx.font = 'bold 15px JetBrains Mono, monospace';
      ctx.fillStyle = '#38bdf8';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('H', baseStation.x, baseStation.y);

      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('BASE DEPOT', baseStation.x, baseStation.y + baseStation.radius + 14);

      // 4. Draw Obstacles
      for (const obs of obstacles) {
        // Safe buffer ring (repulsion zone)
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.arc(obs.x, obs.y, obs.radius + 30, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (obs.type === 'tower') {
          // Antenna Mast
          ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
          ctx.beginPath();
          ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(obs.x, obs.y, obs.radius, 0, Math.PI * 2);
          ctx.stroke();

          // Cross lattice
          ctx.strokeStyle = '#f87171';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(obs.x - obs.radius * 0.7, obs.y - obs.radius * 0.7);
          ctx.lineTo(obs.x + obs.radius * 0.7, obs.y + obs.radius * 0.7);
          ctx.moveTo(obs.x + obs.radius * 0.7, obs.y - obs.radius * 0.7);
          ctx.lineTo(obs.x - obs.radius * 0.7, obs.y + obs.radius * 0.7);
          ctx.stroke();

          // Flashing beacon light
          const beaconAlpha = Math.abs(Math.sin(pulseRef.current * 3));
          ctx.fillStyle = `rgba(239, 68, 68, ${beaconAlpha})`;
          ctx.beginPath();
          ctx.arc(obs.x, obs.y, 4, 0, Math.PI * 2);
          ctx.fill();
        } else if (obs.type === 'ridge') {
          // Rock Ridge / Mountain terrain
          ctx.fillStyle = '#1e293b';
          ctx.strokeStyle = '#475569';
          ctx.lineWidth = 2;
          ctx.beginPath();
          // Draw hexagonal boulder formation
          const sides = 6;
          for (let i = 0; i < sides; i++) {
            const a = (i * Math.PI * 2) / sides;
            const px = obs.x + Math.cos(a) * obs.radius;
            const py = obs.y + Math.sin(a) * obs.radius;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          ctx.strokeStyle = '#334155';
          ctx.beginPath();
          ctx.arc(obs.x, obs.y, obs.radius * 0.5, 0, Math.PI * 2);
          ctx.stroke();
        } else {
          // Structure / Bunker
          ctx.fillStyle = 'rgba(245, 158, 11, 0.15)';
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 2;
          ctx.fillRect(obs.x - obs.radius, obs.y - obs.radius, obs.radius * 2, obs.radius * 2);
          ctx.strokeRect(obs.x - obs.radius, obs.y - obs.radius, obs.radius * 2, obs.radius * 2);
        }

        // Label
        if (obs.label) {
          ctx.font = '9px JetBrains Mono, monospace';
          ctx.fillStyle = '#64748b';
          ctx.textAlign = 'center';
          ctx.fillText(obs.label, obs.x, obs.y + obs.radius + 12);
        }
      }

      // 5. Draw Wildfires
      const windRad = (wind.direction * Math.PI) / 180;
      const windVx = Math.cos(windRad) * wind.speed * 0.4;
      const windVy = Math.sin(windRad) * wind.speed * 0.4;

      for (const fire of fires) {
        if (!fire.isActive || fire.isExtinguished) {
          // Burn scar / extinguished mark
          ctx.fillStyle = 'rgba(71, 85, 105, 0.25)';
          ctx.beginPath();
          ctx.arc(fire.x, fire.y, fire.radius * 0.7, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = '#334155';
          ctx.lineWidth = 1;
          ctx.stroke();
          continue;
        }

        // Spawn smoke/ember particles occasionally
        if (Math.random() < 0.3) {
          particlesRef.current.push({
            x: fire.x + (Math.random() - 0.5) * fire.radius,
            y: fire.y + (Math.random() - 0.5) * fire.radius,
            vx: windVx + (Math.random() - 0.5) * 0.5,
            vy: windVy + (Math.random() - 0.5) * 0.5 - 0.8,
            life: 1.0,
            maxLife: 1.0,
            type: Math.random() < 0.2 ? 'spark' : 'smoke',
            color: Math.random() < 0.2 ? '#f59e0b' : 'rgba(100, 116, 139, 0.4)',
            size: Math.random() * 4 + 3,
          });
        }

        // Multi-layered glowing fire circles
        const fireIntensityNorm = fire.intensity / 100;
        const currentR = fire.radius * (0.85 + pulse * 0.08);

        // Outer heat aura
        const heatGrad = ctx.createRadialGradient(
          fire.x, fire.y, 2,
          fire.x, fire.y, currentR * 1.5
        );
        heatGrad.addColorStop(0, `rgba(239, 68, 68, ${0.4 * fireIntensityNorm})`);
        heatGrad.addColorStop(0.6, `rgba(249, 115, 22, ${0.2 * fireIntensityNorm})`);
        heatGrad.addColorStop(1, 'rgba(239, 68, 68, 0)');

        ctx.fillStyle = heatGrad;
        ctx.beginPath();
        ctx.arc(fire.x, fire.y, currentR * 1.5, 0, Math.PI * 2);
        ctx.fill();

        // Main fire core
        const coreGrad = ctx.createRadialGradient(
          fire.x, fire.y, 0,
          fire.x, fire.y, currentR
        );
        coreGrad.addColorStop(0, '#fef08a'); // white-yellow heat center
        coreGrad.addColorStop(0.35, '#f97316'); // intense flame orange
        coreGrad.addColorStop(0.85, '#dc2626'); // deep red boundary
        coreGrad.addColorStop(1, 'rgba(220, 38, 38, 0)');

        ctx.fillStyle = coreGrad;
        ctx.beginPath();
        ctx.arc(fire.x, fire.y, currentR, 0, Math.PI * 2);
        ctx.fill();

        // Fire perimeter ring
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(fire.x, fire.y, fire.radius, 0, Math.PI * 2);
        ctx.stroke();

        // Intensity % text
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        ctx.fillStyle = '#fef08a';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(fire.intensity)}%`, fire.x, fire.y + 3);
      }

      // 6. Draw Swarm Communication Mesh Lines
      if (showCommMesh) {
        ctx.lineWidth = 1;
        for (let i = 0; i < drones.length; i++) {
          for (let j = i + 1; j < drones.length; j++) {
            const d1 = drones[i];
            const d2 = drones[j];
            const dx = d1.x - d2.x;
            const dy = d1.y - d2.y;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < d1.commRadius) {
              const alpha = Math.max(0, 1 - d / d1.commRadius) * 0.4;
              ctx.strokeStyle = `rgba(56, 189, 248, ${alpha})`;
              ctx.beginPath();
              ctx.moveTo(d1.x, d1.y);
              ctx.lineTo(d2.x, d2.y);
              ctx.stroke();
            }
          }
        }
      }

      // 7. Draw Drone Flight Trails
      if (showTrails) {
        ctx.lineWidth = 1.5;
        for (const drone of drones) {
          if (drone.trail && drone.trail.length > 1) {
            ctx.beginPath();
            ctx.moveTo(drone.trail[0].x, drone.trail[0].y);
            for (let i = 1; i < drone.trail.length; i++) {
              ctx.lineTo(drone.trail[i].x, drone.trail[i].y);
            }
            ctx.strokeStyle = drone.color ? `${drone.color}33` : 'rgba(56, 189, 248, 0.2)';
            ctx.stroke();
          }
        }
      }

      // 8. Draw Drones
      for (const drone of drones) {
        const isSelected = drone.id === selectedDroneId;
        const color = drone.color || '#38bdf8';

        // Sensor / Avoidance Range Ring
        if (showSensorRanges || isSelected) {
          ctx.strokeStyle = isSelected ? 'rgba(56, 189, 248, 0.35)' : 'rgba(148, 163, 184, 0.12)';
          ctx.setLineDash([2, 4]);
          ctx.beginPath();
          ctx.arc(drone.x, drone.y, drone.sensorRadius, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Near-Miss / Safety Halo
        if (drone.nearMissCount > 0) {
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.8)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(drone.x, drone.y, drone.radius + 6, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Active Water Suppression Spray Animation
        if (drone.state === 'SUPPRESSING') {
          // Emit water droplets towards targeted fire
          for (let s = 0; s < 3; s++) {
            const sprayAngle = (drone.heading || 0) + (Math.random() - 0.5) * 0.8;
            particlesRef.current.push({
              x: drone.x,
              y: drone.y,
              vx: Math.cos(sprayAngle) * (Math.random() * 3 + 2),
              vy: Math.sin(sprayAngle) * (Math.random() * 3 + 2),
              life: 1.0,
              maxLife: 1.0,
              type: 'water',
              color: 'rgba(56, 189, 248, 0.8)',
              size: Math.random() * 3 + 2,
            });
          }

          // Blue mist cone
          ctx.fillStyle = 'rgba(14, 165, 233, 0.25)';
          ctx.beginPath();
          ctx.arc(drone.x, drone.y, 22, 0, Math.PI * 2);
          ctx.fill();
        }

        // Drone Quadcopter Body
        ctx.save();
        ctx.translate(drone.x, drone.y);

        // Heading angle (radians)
        const heading = Math.atan2(drone.vy, drone.vx) || 0;
        ctx.rotate(heading);

        // Rotor arms (diagonal cross)
        ctx.strokeStyle = isSelected ? '#ffffff' : color;
        ctx.lineWidth = 2;
        const arm = 9;
        ctx.beginPath();
        ctx.moveTo(-arm, -arm);
        ctx.lineTo(arm, arm);
        ctx.moveTo(-arm, arm);
        ctx.lineTo(arm, -arm);
        ctx.stroke();

        // 4 Spinning Rotors
        const rotorAngle = pulseRef.current * 8;
        const rotorR = 4.5;
        const rotorOffsets = [
          [-arm, -arm],
          [arm, -arm],
          [arm, arm],
          [-arm, arm],
        ];

        ctx.strokeStyle = isSelected ? '#38bdf8' : 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = 1.2;
        for (const [rx, ry] of rotorOffsets) {
          ctx.beginPath();
          ctx.arc(rx, ry, rotorR, 0, Math.PI * 2);
          ctx.stroke();

          // rotor blade line
          ctx.beginPath();
          ctx.moveTo(rx + Math.cos(rotorAngle) * rotorR, ry + Math.sin(rotorAngle) * rotorR);
          ctx.lineTo(rx - Math.cos(rotorAngle) * rotorR, ry - Math.sin(rotorAngle) * rotorR);
          ctx.stroke();
        }

        // Center fuselage
        ctx.fillStyle = drone.state === 'RETURNING' ? '#f59e0b' : color;
        ctx.beginPath();
        ctx.arc(0, 0, 5, 0, Math.PI * 2);
        ctx.fill();

        // Forward nose pointer
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.moveTo(7, 0);
        ctx.lineTo(2, -3);
        ctx.lineTo(2, 3);
        ctx.closePath();
        ctx.fill();

        ctx.restore();

        // Water level arc (encircles the drone: 0 to 100%)
        const waterRatio = Math.max(0, drone.water / drone.maxWater);
        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(drone.x, drone.y, 14, -Math.PI / 2, -Math.PI / 2 + waterRatio * Math.PI * 2);
        ctx.stroke();

        // Battery level bar
        const barW = 20;
        const barH = 3;
        const bx = drone.x - barW / 2;
        const by = drone.y + 17;

        ctx.fillStyle = '#0f172a';
        ctx.fillRect(bx, by, barW, barH);
        const battColor = drone.battery > 50 ? '#22c55e' : drone.battery > 20 ? '#f59e0b' : '#ef4444';
        ctx.fillStyle = battColor;
        ctx.fillRect(bx, by, barW * (drone.battery / 100), barH);

        // Drone Label / Callout
        ctx.font = 'bold 9px JetBrains Mono, monospace';
        ctx.fillStyle = isSelected ? '#ffffff' : '#cbd5e1';
        ctx.textAlign = 'center';
        ctx.fillText(drone.id, drone.x, drone.y - 18);

        if (drone.state === 'RETURNING' || drone.isManualRTB) {
          ctx.fillStyle = '#f59e0b';
          ctx.font = '8px JetBrains Mono, monospace';
          ctx.fillText('RTB', drone.x, drone.y - 28);
        } else if (drone.state === 'REFILLING') {
          ctx.fillStyle = '#22c55e';
          ctx.font = '8px JetBrains Mono, monospace';
          ctx.fillText('REFUEL', drone.x, drone.y - 28);
        }
      }

      // 9. Update & Draw Active Particles (Smoke, Water spray, Sparks)
      const currentParticles = particlesRef.current;
      for (let i = currentParticles.length - 1; i >= 0; i--) {
        const p = currentParticles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.life -= 0.035;

        if (p.life <= 0) {
          currentParticles.splice(i, 1);
          continue;
        }

        const alpha = p.life / p.maxLife;
        ctx.fillStyle = p.type === 'water'
          ? `rgba(56, 189, 248, ${alpha})`
          : p.type === 'spark'
          ? `rgba(251, 191, 36, ${alpha})`
          : `rgba(148, 163, 184, ${alpha * 0.4})`;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (p.type === 'smoke' ? (2 - alpha) : alpha), 0, Math.PI * 2);
        ctx.fill();
      }

      // 10. Wind Compass HUD in top right
      const compassX = width - 48;
      const compassY = 48;
      const compassR = 24;

      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(compassX, compassY, compassR, 0, Math.PI * 2);
      ctx.stroke();

      // Cardinal N
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('N', compassX, compassY - compassR + 7);

      // Wind Arrow
      const arrowLen = 16;
      const ax = compassX + Math.cos(windRad) * arrowLen;
      const ay = compassY + Math.sin(windRad) * arrowLen;

      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(compassX, compassY);
      ctx.lineTo(ax, ay);
      ctx.stroke();

      // Arrow head
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(ax, ay, 3, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = '8px JetBrains Mono, monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`${wind.speed.toFixed(1)} m/s`, compassX, compassY + compassR + 10);

      // Selected Tool indicator banner on bottom-left
      if (toolMode !== 'inspect') {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1;
        ctx.fillRect(16, height - 36, 220, 24);
        ctx.strokeRect(16, height - 36, 220, 24);

        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillStyle = '#38bdf8';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const label = toolMode === 'add_fire'
          ? 'MODE: CLICK TO IGNITE FIRE'
          : toolMode === 'add_obstacle'
          ? 'MODE: CLICK TO PLACE TOWER'
          : 'MODE: CLICK FOR WATER DUMP';
        ctx.fillText(label, 26, height - 24);
      }

      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      isMounted = false;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [drones, fires, obstacles, baseStation, wind, selectedDroneId, toolMode, showCommMesh, showSensorRanges, showTrails]);

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const clickCanvasX = (e.clientX - rect.left) * dpr;
    const clickCanvasY = (e.clientY - rect.top) * dpr;

    const { scale, offsetX, offsetY } = transformRef.current;
    const clickX = (clickCanvasX - offsetX) / scale;
    const clickY = (clickCanvasY - offsetY) / scale;

    if (clickX < 0 || clickX > 960 || clickY < 0 || clickY > 640) return;

    if (toolMode === 'inspect') {
      // Find closest drone within 24 units
      let foundDroneId: string | null = null;
      for (const d of drones) {
        const dx = d.x - clickX;
        const dy = d.y - clickY;
        if (Math.sqrt(dx * dx + dy * dy) <= 24) {
          foundDroneId = d.id;
          break;
        }
      }
      onSelectDrone(foundDroneId);
    } else {
      onCanvasClick({ x: clickX, y: clickY });
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex items-center justify-center bg-slate-950 overflow-hidden border border-slate-800/80 rounded-xl shadow-2xl"
    >
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        className={`w-full h-full block ${
          toolMode !== 'inspect' ? 'cursor-crosshair' : 'cursor-pointer'
        }`}
        data-testid="tactical-radar-canvas"
      />
    </div>
  );
};
