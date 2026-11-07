
'''
    Needs px4 to run
    https://px4.io/software/getting-started/hello-sky/ 
    

'''

import asyncio

from mavsdk.asyncio import Mavsdk, Configuration, ComponentType
from mavsdk.asyncio.plugins.action import ActionAsync
from mavsdk.asyncio.plugins.telemetry import TelemetryAsync


async def main():

    # Initialize MAVSDK
    config = Configuration.create_with_component_type(
        ComponentType.GROUND_STATION
    )

    sdk = Mavsdk(config)

    # Connect to PX4 SITL
    print("Connecting to PX4 SITL...")

    await sdk.add_any_connection(
        "udpin://0.0.0.0:14540"
    )

    # Discover autopilot
    print("Waiting for PX4...")

    drone = await sdk.first_autopilot(30.0)

    if drone is None:
        print("ERROR: No PX4 autopilot detected.")
        return

    print("Connected to PX4!")

    # Initialize plugins
    action = ActionAsync(drone)
    telemetry = TelemetryAsync(drone)

    # Wait for position estimate
    print("Waiting for position estimate...")

    async for health in telemetry.subscribe_health():
        if (
            health.is_global_position_ok
            and health.is_home_position_ok
        ):
            print("Position estimate ready.")
            break

    try:
        # Takeoff
        print("Starting simulated takeoff...")

        await action.arm()
        await action.takeoff()

        print("Takeoff command accepted.")

        # Monitor telemetry during the simulation
        print("Monitoring simulated vehicle...")

        async def monitor_altitude():
            async for position in telemetry.subscribe_position():
                print(
                    f"Altitude: "
                    f"{position.relative_altitude_m:.2f} m"
                )

        monitor_task = asyncio.create_task(monitor_altitude())

        # Hover period
        await asyncio.sleep(10)

        # Stop telemetry task
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Land
        print("Starting simulated landing...")

        await action.land()

        print("Landing command accepted.")

    except Exception as error:
        print(f"Command failed: {error}")

        # Do not automatically disarm or issue another command.
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted.")
