import asyncio

from mavsdk.asyncio import Mavsdk, Configuration, ComponentType
from mavsdk.asyncio.plugins.telemetry import TelemetryAsync


async def main():

    configuration = Configuration.create_with_component_type(
        ComponentType.GROUND_STATION
    )

    sdk = Mavsdk(configuration)

    print("Listening on UDP port 14540...")

    await sdk.add_any_connection(
        "udpin://0.0.0.0:14540"
    )

    print("Waiting for PX4...")

    drone = await sdk.first_autopilot(30.0)

    if drone is None:
        print("No PX4 autopilot detected.")
        return

    print("PX4 connected!")

    telemetry = TelemetryAsync(drone)

    async for position in telemetry.subscribe_position():
        print(
            f"Latitude: {position.latitude_deg:.6f}, "
            f"Longitude: {position.longitude_deg:.6f}, "
            f"Altitude: {position.relative_altitude_m:.2f} m"
        )


if __name__ == "__main__":
    asyncio.run(main())