
import asyncio
import random
import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Drone:
    drone_id: str
    latitude: float
    longitude: float
    altitude: float
    battery: float
    connected: bool = True
    last_update: str = ""


# Synthetic demonstration vehicles
drones = [
    Drone("DRONE-01", 47.3977, 8.5456, 0.0, 100.0),
    Drone("DRONE-02", 47.3980, 8.5460, 0.0, 96.0),
    Drone("DRONE-03", 47.3975, 8.5465, 0.0, 92.0),
]


async def simulate_telemetry(drone):
    """Generate demonstration telemetry for one vehicle."""

    while True:
        drone.latitude += random.uniform(-0.00001, 0.00001)
        drone.longitude += random.uniform(-0.00001, 0.00001)
        drone.altitude += random.uniform(-0.02, 0.02)

        drone.altitude = max(0, drone.altitude)

        drone.battery = max(0, drone.battery - 0.01)

        drone.last_update = datetime.now().strftime("%H:%M:%S")

        await asyncio.sleep(1)


def display_dashboard():

    os.system("cls" if os.name == "nt" else "clear")

    print("=" * 85)
    print("               MULTI-DRONE TELEMETRY DASHBOARD")
    print("=" * 85)

    print(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
    print()

    print(
        f"{'ID':<12}"
        f"{'STATUS':<12}"
        f"{'LATITUDE':<17}"
        f"{'LONGITUDE':<17}"
        f"{'ALTITUDE':<13}"
        f"{'BATTERY':<12}"
    )

    print("-" * 85)

    for drone in drones:

        status = "CONNECTED" if drone.connected else "OFFLINE"

        print(
            f"{drone.drone_id:<12}"
            f"{status:<12}"
            f"{drone.latitude:<17.6f}"
            f"{drone.longitude:<17.6f}"
            f"{drone.altitude:<13.2f}"
            f"{drone.battery:<11.1f}%"
        )

    print("-" * 85)
    print(f"Total vehicles: {len(drones)}")
    print(f"Connected: {sum(d.connected for d in drones)}")
    print(f"Offline: {sum(not d.connected for d in drones)}")
    print("=" * 85)


async def dashboard_loop():

    while True:
        display_dashboard()
        await asyncio.sleep(1)


async def main():

    print("Starting multi-drone telemetry demonstration...")

    tasks = [
        simulate_telemetry(drone)
        for drone in drones
    ]

    tasks.append(dashboard_loop())

    await asyncio.gather(*tasks)


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nDashboard stopped.")
