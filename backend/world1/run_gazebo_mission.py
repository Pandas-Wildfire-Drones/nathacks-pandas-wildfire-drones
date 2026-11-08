#!/usr/bin/env python3
"""
================================================================================
PX4 + GAZEBO SITL AUTONOMOUS MISSION RUNNER
================================================================================
File: run_gazebo_mission.py

This script:
 1. Connects to PX4 Autopilot running in Gazebo SITL (udp://:14540)
 2. Arms motors and autonomously takes off to 15m
 3. Streams telemetry (Position NED, Attitude, LIDAR ranges)
 4. Calls YOUR custom algorithm from 'my_avoidance_algorithm.py'
 5. Streams velocity setpoints to PX4 Offboard Mode
 6. Actuates fire suppression payload when algorithm flags drop trigger
 7. Safely executes Return-To-Launch (RTL) upon mission completion
================================================================================
"""

import asyncio
import math
import asyncio
import math
import sys

# Reliable MAVSDK imports
try:
    from mavsdk_grpc import System
    from mavsdk_grpc.offboard import (OffboardError, VelocityNedYaw)
except ImportError:
    from mavsdk import System
    from mavsdk.offboard import (OffboardError, VelocityNedYaw)

from my_avoidance_algorithm import MyAutonomousAlgorithm
# Import YOUR custom algorithm class from the template file
from my_avoidance_algorithm import MyAutonomousAlgorithm

# Mission Configuration
CONNECTION_URL = "udp://:14540"
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
        self.drone = System()
        # Instantiate your custom algorithm!
        self.algorithm = MyAutonomousAlgorithm(
            safety_margin_m=SAFETY_MARGIN,
            max_speed_mps=CRUISE_SPEED
        )
        self.current_pos_ned = (0.0, 0.0, 0.0)
        self.current_attitude = (0.0, 0.0, 0.0)
        self.forward_dist = 25.0
        self.lidar_ranges = [25.0] * 16
        self.is_connected = False

    async def connect_to_gazebo(self):
        """Connects to PX4 Gazebo SITL on UDP 14540"""
        print(f"[*] Connecting to PX4 Gazebo SITL on {CONNECTION_URL}...")
        await self.drone.connect(system_address=CONNECTION_URL)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"[+] Connected to PX4 in Gazebo! System UUID: {state}")
                self.is_connected = True
                break

    async def start_telemetry(self):
        """Streams real-time position, attitude, and simulated LIDAR distance sensor"""
        asyncio.create_task(self._listen_position())
        asyncio.create_task(self._listen_attitude())
        asyncio.create_task(self._listen_distance_sensor())

    async def _listen_position(self):
        async for pos in self.drone.telemetry.position_velocity_ned():
            p = pos.position
            self.current_pos_ned = (p.north_m, p.east_m, p.down_m)

    async def _listen_attitude(self):
        async for att in self.drone.telemetry.attitude_euler():
            self.current_attitude = (att.roll_deg, att.pitch_deg, att.yaw_deg)

    async def _listen_distance_sensor(self):
        async for dist in self.drone.telemetry.distance_sensor():
            self.forward_dist = dist.current_distance_m

    async def takeoff(self):
        """Arms drone and takes off to cruise altitude"""
        print("[*] Waiting for GPS lock in Gazebo...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("[+] GPS 3D fix locked in Gazebo world.")
                break

        print("[*] Arming motors...")
        await self.drone.action.arm()
        print(f"[*] Taking off to {CRUISE_ALTITUDE}m...")
        await self.drone.action.set_takeoff_altitude(CRUISE_ALTITUDE)
        await self.drone.action.takeoff()
        await asyncio.sleep(8)
        print("[+] Takeoff complete. Ready for autonomous offboard control.")

    async def run_autonomous_loop(self):
        """Main offboard loop feeding data into YOUR algorithm at 10 Hz"""
        # Send initial setpoint before starting offboard mode
        await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        try:
            await self.drone.offboard.start()
            print("[+] PX4 OFFBOARD MODE ACTIVE. Handing flight control to YOUR algorithm!")
        except OffboardError as e:
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
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, yaw))

                # Handle payload drop trigger
                if trigger_drop:
                    print(f"[+] FIRE TARGET LOCK: Deploying water drop payload on {t_id}!")
                    await self.deploy_payload()
                    break

                # Check if arrived at goal
                if vn == 0.0 and ve == 0.0:
                    print(f"[+] Reached position above {t_id}!")
                    break

                await asyncio.sleep(0.1) # 10 Hz loop

        print("\n[+] All targets reached. Initiating Return-To-Launch (RTL)...")
        await self.drone.offboard.stop()
        await self.drone.action.return_to_launch()

    async def deploy_payload(self):
        """Triggers the payload release actuator via MAVLink servo command"""
        try:
            # Set Actuator #1 to 100% open
            await self.drone.action.set_actuator(1, 1.0)
            await asyncio.sleep(2.5)
            await self.drone.action.set_actuator(1, 0.0)
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
