#!/usr/bin/env python3
"""
================================================================================
PX4 + GAZEBO SITL AUTONOMOUS MISSION RUNNER (MAVSDK 4 NATIVE)
================================================================================
File: run_gazebo_mission.py
"""

import asyncio
import math
import sys

# MAVSDK 4 Native vs Legacy gRPC Import Resolver
IS_MAVSDK_V4 = False

try:
    # 1. Native MAVSDK v4 (pip install mavsdk >= 4.0)
    from mavsdk.asyncio import Mavsdk, Configuration, ComponentType
    from mavsdk.asyncio.plugins.action import ActionAsync
    from mavsdk.asyncio.plugins.offboard import OffboardAsync, VelocityNedYaw
    from mavsdk.asyncio.plugins.telemetry import TelemetryAsync
    IS_MAVSDK_V4 = True
    print("[+] Detected Native MAVSDK v4.0+")
except Exception:
    try:
        from mavsdk import System
        from mavsdk.offboard import (OffboardError, VelocityNedYaw)
        IS_MAVSDK_V4 = False
    except Exception:
        import mavsdk_grpc as mavsdk
        from mavsdk_grpc import System
        from mavsdk_grpc.offboard import (OffboardError, VelocityNedYaw)
        IS_MAVSDK_V4 = False

# Import YOUR custom algorithm class from my_avoidance_algorithm.py
from my_avoidance_algorithm import MyAutonomousAlgorithm

# Mission Configuration
CONNECTION_URL = "udpin://0.0.0.0:14540"
CRUISE_ALTITUDE = 15.0
CRUISE_SPEED = 6.0
SAFETY_MARGIN = 6.0

# Wildfire Hotspots in the Gazebo Obstacle World (North, East, Down in meters)
WILDFIRE_TARGETS = [
    {"id": "Hotspot_Alpha", "pos": (48.0, 8.0, -CRUISE_ALTITUDE)},
    {"id": "Hotspot_Bravo", "pos": (75.0, -8.0, -CRUISE_ALTITUDE)},
    {"id": "Hotspot_Charlie", "pos": (105.0, 12.0, -CRUISE_ALTITUDE)},
]

class GazeboMissionRunner:
    def __init__(self):
        self.algorithm = MyAutonomousAlgorithm(
            safety_margin_m=SAFETY_MARGIN,
            max_speed_mps=CRUISE_SPEED
        )
        self.current_pos_ned = (0.0, 0.0, 0.0)
        self.current_attitude = (0.0, 0.0, 0.0)
        self.forward_dist = 25.0
        self.lidar_ranges = [25.0] * 16

        self.mavsdk = None
        self.drone = None
        self.action = None
        self.offboard = None
        self.telemetry = None

    async def connect_to_gazebo(self):
        """Connects to PX4 Gazebo SITL on UDP 14540"""
        print(f"[*] Initializing connection to PX4 Gazebo SITL on {CONNECTION_URL}...")

        if IS_MAVSDK_V4:
            config = Configuration.create_with_component_type(ComponentType.GROUND_STATION)
            self.mavsdk = Mavsdk(config)
            await self.mavsdk.add_any_connection(CONNECTION_URL)

            print("[*] Waiting for PX4 autopilot in Gazebo (timeout 15s)...")
            self.drone = await self.mavsdk.first_autopilot(15.0)
            if self.drone is None:
                print("[!] Timeout: No PX4 autopilot detected in Gazebo SITL.")
                sys.exit(1)

            print("[+] Successfully discovered PX4 Autopilot in Gazebo!")
            self.action = ActionAsync(self.drone)
            self.offboard = OffboardAsync(self.drone)
            self.telemetry = TelemetryAsync(self.drone)
        else:
            self.drone = System()
            await self.drone.connect(system_address="udp://:14540")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    break
            self.action = self.drone.action
            self.offboard = self.drone.offboard
            self.telemetry = self.drone.telemetry

    async def start_telemetry(self):
        """Streams real-time position, attitude, and distance sensor to algorithm"""
        asyncio.create_task(self._telemetry_poll_loop())

    async def _telemetry_poll_loop(self):
        """Polls telemetry at 20 Hz without crashing on MAVSDK v4 coroutines"""
        while True:
            try:
                # 1. Position
                pos_call = self.telemetry.position_velocity_ned()
                pos = await pos_call if asyncio.iscoroutine(pos_call) else pos_call
                p = getattr(pos, 'position', pos)
                self.current_pos_ned = (
                    getattr(p, 'north_m', self.current_pos_ned[0]),
                    getattr(p, 'east_m', self.current_pos_ned[1]),
                    getattr(p, 'down_m', self.current_pos_ned[2])
                )
            except Exception:
                pass

            try:
                # 2. Attitude
                att_call = self.telemetry.attitude_euler()
                att = await att_call if asyncio.iscoroutine(att_call) else att_call
                self.current_attitude = (
                    getattr(att, 'roll_deg', self.current_attitude[0]),
                    getattr(att, 'pitch_deg', self.current_attitude[1]),
                    getattr(att, 'yaw_deg', self.current_attitude[2])
                )
            except Exception:
                pass

            try:
                # 3. Distance Sensor
                dist_call = self.telemetry.distance_sensor()
                dist = await dist_call if asyncio.iscoroutine(dist_call) else dist_call
                self.forward_dist = getattr(dist, 'current_distance_m', self.forward_dist)
            except Exception:
                pass

            await asyncio.sleep(0.05)  # 20 Hz update rate

    async def takeoff(self):
        """Arms drone and takes off to cruise altitude"""
        print("[*] Waiting for PX4 EKF2 estimator and GPS to converge in Gazebo...")
        await asyncio.sleep(3)  # Give EKF2 3 seconds to boot

        armed = False
        print("[*] Arming motors (retrying until preflight checks pass)...")
        for attempt in range(1, 26):
            try:
                await self.action.arm()
                armed = True
                print("[+] Motors armed successfully!")
                break
            except Exception:
                print(f"[*] Pre-flight estimator settling... (Attempt {attempt}/25)")
                await asyncio.sleep(1.5)

        if not armed:
            print("[!] Arming denied. In the PX4 terminal, type 'commander check' to see which check is failing.")
            print("[!] Tip: You can bypass pre-flight checks in PX4 terminal with: commander arm -f")
            return

        print(f"[*] Taking off to {CRUISE_ALTITUDE}m...")
        await self.action.set_takeoff_altitude(CRUISE_ALTITUDE)
        await self.action.takeoff()
        await asyncio.sleep(8)
        print("[+] Takeoff complete. Ready for autonomous offboard control.")

    async def run_autonomous_loop(self):
        """Main offboard loop feeding data into YOUR algorithm at 10 Hz"""
        await self.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await self.offboard.start()
            print("[+] PX4 OFFBOARD MODE ACTIVE. Handing flight control to YOUR algorithm!")
        except Exception as e:
            print(f"[!] Offboard error: {e}")
            return

        for target in WILDFIRE_TARGETS:
            t_id = target["id"]
            t_pos = target["pos"]
            print(f"\n[*] Routing to {t_id} at {t_pos}...")

            while True:
                # CALL YOUR CUSTOM ALGORITHM
                vn, ve, vd, yaw, trigger_drop = self.algorithm.compute_control_step(
                    current_pos_ned=self.current_pos_ned,
                    current_attitude=self.current_attitude,
                    lidar_ranges=self.lidar_ranges,
                    forward_obstacle_dist=self.forward_dist,
                    target_goal_ned=t_pos,
                    is_fire_hotspot=True
                )

                # Send velocity command to Gazebo simulated drone
                await self.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, yaw))

                # Handle payload drop trigger
                if trigger_drop:
                    print(f"[+] FIRE TARGET LOCK: Deploying water drop payload on {t_id}!")
                    await self.deploy_payload()
                    break

                if vn == 0.0 and ve == 0.0:
                    print(f"[+] Reached position above {t_id}!")
                    break

                await asyncio.sleep(0.1)  # 10 Hz loop

        print("\n[+] All targets reached. Initiating Return-To-Launch (RTL)...")
        try:
            await self.offboard.stop()
            await self.action.return_to_launch()
        except Exception as e:
            print(f"[*] RTL status: {e}")

    async def deploy_payload(self):
        """Triggers the payload release actuator via MAVLink servo command"""
        try:
            await self.action.set_actuator(1, 1.0)
            await asyncio.sleep(2.5)
            await self.action.set_actuator(1, 0.0)
        except Exception as err:
            print(f"[*] Payload actuator event: {err}")
            await asyncio.sleep(2)

async def main():
    runner = GazeboMissionRunner()
    await runner.connect_to_gazebo()
    await runner.start_telemetry()
    await runner.takeoff()
    await runner.run_autonomous_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[!] Emergency stop requested.")
        sys.exit(0)