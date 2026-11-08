import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

from mavsdk.asyncio import Mavsdk, Configuration, ComponentType
from mavsdk.asyncio.plugins.telemetry import TelemetryAsync


# PX4 SITL instance 1 sends its MAVSDK stream to UDP 14541.
# PX4 SITL instance 2 sends its MAVSDK stream to UDP 14542.
LISTEN_PORTS = (14541, 14542)
EXPECTED_VEHICLES = len(LISTEN_PORTS)
DISCOVERY_TIMEOUT_SECONDS = 45


@dataclass
class VehicleRow:
    label: str
    system_id: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    battery_percent: Optional[float] = None
    global_position_ok: Optional[bool] = None
    last_position_time: Optional[float] = None
    error: Optional[str] = None


def fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if value is not None else "--"


async def read_position(telemetry: TelemetryAsync, row: VehicleRow) -> None:
    try:
        async for position in telemetry.subscribe_position():
            row.latitude = position.latitude_deg
            row.longitude = position.longitude_deg
            row.altitude_m = position.relative_altitude_m
            row.last_position_time = time.monotonic()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        row.error = f"Position telemetry: {exc}"


async def read_battery(telemetry: TelemetryAsync, row: VehicleRow) -> None:
    try:
        async for battery in telemetry.subscribe_battery():
            value = battery.remaining_percent
            # MAVLink may report an unknown battery percentage as a negative value.
            row.battery_percent = value * 100.0 if value is not None and value >= 0 else None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        row.error = f"Battery telemetry: {exc}"


async def read_health(telemetry: TelemetryAsync, row: VehicleRow) -> None:
    try:
        async for health in telemetry.subscribe_health():
            row.global_position_ok = health.is_global_position_ok
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        row.error = f"Health telemetry: {exc}"


async def discover_vehicles(sdk: Mavsdk):
    deadline = time.monotonic() + DISCOVERY_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        systems = await sdk.get_systems()
        autopilots = [system for system in systems if system.has_autopilot()]

        if len(autopilots) >= EXPECTED_VEHICLES:
            return autopilots[:EXPECTED_VEHICLES]

        await asyncio.sleep(1)

    systems = await sdk.get_systems()
    autopilots = [system for system in systems if system.has_autopilot()]
    return autopilots


async def dashboard_loop(rows: list[VehicleRow]) -> None:
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("PX4 SITL MULTI-VEHICLE TELEMETRY MONITOR")
        print("Read-only: this program sends no vehicle commands.")
        print(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 112)
        print(
            f"{'VEHICLE':<12} {'SYS ID':<8} {'STATUS':<12} "
            f"{'LATITUDE':>13} {'LONGITUDE':>13} {'ALT (m)':>10} "
            f"{'BATTERY':>10} {'GLOBAL POS':>12}"
        )
        print("-" * 112)

        now = time.monotonic()
        for row in rows:
            age = now - row.last_position_time if row.last_position_time is not None else None
            status = "LIVE" if age is not None and age < 5 else "WAITING"
            global_pos = (
                "OK" if row.global_position_ok is True
                else "NO" if row.global_position_ok is False
                else "--"
            )
            battery = f"{row.battery_percent:.1f}%" if row.battery_percent is not None else "--"

            print(
                f"{row.label:<12} {row.system_id:<8} {status:<12} "
                f"{fmt(row.latitude, '.6f'):>13} "
                f"{fmt(row.longitude, '.6f'):>13} "
                f"{fmt(row.altitude_m, '.2f'):>10} "
                f"{battery:>10} {global_pos:>12}"
            )
            if row.error:
                print(f"  Note: {row.error}")

        print("-" * 112)
        print(f"Vehicles discovered: {len(rows)} / {EXPECTED_VEHICLES}")
        print("Press Ctrl+C to stop the monitor.")
        await asyncio.sleep(1)


async def main() -> None:
    sdk = Mavsdk(
        Configuration.create_with_component_type(ComponentType.GROUND_STATION)
    )

    # Listen on the per-instance MAVSDK UDP ports.
    for port in LISTEN_PORTS:
        print(f"Opening MAVSDK UDP listener on {port}...")
        await sdk.add_any_connection(f"udpin://0.0.0.0:{port}")

    print(f"Waiting up to {DISCOVERY_TIMEOUT_SECONDS}s for {EXPECTED_VEHICLES} PX4 SITL vehicles...")
    systems = await discover_vehicles(sdk)

    if not systems:
        print("No PX4 autopilots detected.")
        print("Check that PX4 SITL is running and that its API links target ports 14541 and 14542.")
        return

    if len(systems) < EXPECTED_VEHICLES:
        print(f"Detected {len(systems)} of {EXPECTED_VEHICLES} expected vehicle(s). Showing those discovered.")

    rows: list[VehicleRow] = []
    telemetry_tasks = []

    for index, system in enumerate(systems, start=1):
        system_id = await system.get_system_id()
        row = VehicleRow(label=f"PX4-{index}", system_id=system_id)
        rows.append(row)

        telemetry = TelemetryAsync(system)
        telemetry_tasks.extend(
            [
                asyncio.create_task(read_position(telemetry, row)),
                asyncio.create_task(read_battery(telemetry, row)),
                asyncio.create_task(read_health(telemetry, row)),
            ]
        )

    try:
        await dashboard_loop(rows)
    finally:
        for task in telemetry_tasks:
            task.cancel()
        await asyncio.gather(*telemetry_tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
