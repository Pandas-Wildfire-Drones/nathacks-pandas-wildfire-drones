#!/usr/bin/env python3
"""
================================================================================
USER AUTONOMOUS ALGORITHM TEMPLATE (Gazebo SITL + PX4 MAVSDK)
================================================================================
File: my_avoidance_algorithm.py

INSERT YOUR OWN ALGORITHM HERE!
You can implement any guidance method:
 - Artificial Potential Field (APF)
 - Vector Field Histogram (VFH / VFH+)
 - Rapidly-exploring Random Tree (RRT / RRT*)
 - Model Predictive Control (MPC)
 - Deep Reinforcement Learning / Neural Policy
================================================================================
"""

import math
from typing import List, Tuple, Dict, Any

class MyAutonomousAlgorithm:
    def __init__(self, safety_margin_m: float = 6.0, max_speed_mps: float = 5.0):
        """
        Initialize your algorithm parameters, state buffers, or weights here.
        """
        self.safety_margin = safety_margin_m
        self.max_speed = max_speed_mps
        print(f"[Algorithm] Initialized with safety margin: {self.safety_margin}m, max speed: {self.max_speed}m/s")

    def compute_control_step(
        self,
        current_pos_ned: Tuple[float, float, float],    # (North_m, East_m, Down_m)
        current_attitude: Tuple[float, float, float],   # (Roll_deg, Pitch_deg, Yaw_deg)
        lidar_ranges: List[float],                      # 16-beam 360° LIDAR ranges in meters
        forward_obstacle_dist: float,                   # Distance to nearest obstacle ahead in meters
        target_goal_ned: Tuple[float, float, float],    # (Target_North, Target_East, Target_Down)
        is_fire_hotspot: bool = False                   # True if currently tracking a wildfire
    ) -> Tuple[float, float, float, float, bool]:
        """
        CALLED EVERY 100ms (10 Hz) DURING FLIGHT.
        
        Args:
            current_pos_ned: Current drone position in PX4 local NED coordinates (meters).
            current_attitude: Current drone attitude (Roll, Pitch, Yaw) in degrees.
            lidar_ranges: Array of distance measurements around the drone (0° to 360°).
            forward_obstacle_dist: Minimum distance reading from forward range sensor.
            target_goal_ned: Desired destination waypoint (North, East, Down).
            is_fire_hotspot: Flag indicating if this goal is a wildfire to extinguish.
            
        Returns:
            Tuple of:
            - vel_north (float): Commanded velocity North in m/s
            - vel_east  (float): Commanded velocity East in m/s
            - vel_down  (float): Commanded velocity Down in m/s (negative is CLIMBING)
            - yaw_deg   (float): Desired heading in degrees (0 = North, 90 = East)
            - trigger_water_drop (bool): True when aligned over fire to actuate payload drop
        """
        curr_n, curr_e, curr_d = current_pos_ned
        goal_n, goal_e, goal_d = target_goal_ned
        curr_yaw = current_attitude[2]

        # ---------------------------------------------------------------------
        # >>> YOUR CUSTOM ALGORITHM STARTS HERE <<<
        # ---------------------------------------------------------------------

        # Example: Vector to target goal
        dn = goal_n - curr_n
        de = goal_e - curr_e
        dist_to_goal = math.hypot(dn, de)

        if dist_to_goal < 1.0:
            # Reached goal
            trigger_water = is_fire_hotspot
            return 0.0, 0.0, 0.0, curr_yaw, trigger_water

        # 1. Attractive goal-seeking component
        att_n = (dn / dist_to_goal) * self.max_speed
        att_e = (de / dist_to_goal) * self.max_speed

        # 2. Obstacle Avoidance Repulsion
        rep_n = 0.0
        rep_e = 0.0

        if forward_obstacle_dist < self.safety_margin:
            # Obstacle detected! Push perpendicular to clear the obstruction
            penetration = self.safety_margin - forward_obstacle_dist
            rep_strength = penetration * 3.5

            # Lateral detour angle
            yaw_rad = math.radians(curr_yaw)
            rep_n = -math.sin(yaw_rad) * rep_strength
            rep_e = math.cos(yaw_rad) * rep_strength

        # 3. Combine resultant velocities
        cmd_n = max(-self.max_speed, min(self.max_speed, att_n + rep_n))
        cmd_e = max(-self.max_speed, min(self.max_speed, att_e + rep_e))

        # Altitude hold at target height (maintain goal_d, e.g. -15m)
        alt_error = goal_d - curr_d
        cmd_d = max(-1.5, min(1.5, alt_error * 0.8))

        # Face towards movement direction
        target_yaw = math.degrees(math.atan2(cmd_e, cmd_n))
        if target_yaw < 0:
            target_yaw += 360.0

        # Check if directly overhead wildfire for suppression trigger
        trigger_water = False
        if is_fire_hotspot and dist_to_goal < 3.0:
            trigger_water = True

        # ---------------------------------------------------------------------
        # >>> YOUR CUSTOM ALGORITHM ENDS HERE <<<
        # ---------------------------------------------------------------------

        return cmd_n, cmd_e, cmd_d, target_yaw, trigger_water
