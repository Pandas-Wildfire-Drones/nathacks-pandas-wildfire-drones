# Accord Autonomy Engineering Decision Record

## Purpose

This document records the safety hierarchy, autonomy decisions, assumptions,
tradeoffs, and test evidence for the per-drone autonomy subsystem.

The autonomy subsystem is responsible for individual-drone decisions such as:

- coverage search
- path planning
- path deconfliction
- resource assessment
- water-drop safety
- health/failsafe decisions
- path claims
- refill decisions
- mission execution

The swarm brain, perception system, simulator/backend, and ground operations
remain separate ownership areas.

---

## 1. Safety Hierarchy

Autonomy follows this priority order:

1. **Hard safety gates**
   - drone health
   - communication health
   - collision/separation constraints
   - valid geometry
   - lost/fault states

2. **Mission validity**
   - valid task
   - correct assignment
   - current epoch
   - valid target
   - valid target state

3. **Resource safety**
   - battery reserve
   - fuel reserve
   - water availability
   - sufficient endurance

4. **Path/deconfliction**
   - route feasibility
   - time-window validity
   - separation from other drones
   - altitude separation where supplied

5. **Mission objective**
   - navigate
   - scan
   - extinguish
   - verify
   - refill

A lower-priority mission objective must never override a higher-priority
safety constraint.

---

## 2. Drone Health

### Decision

A healthy drone may continue.

An explicitly failed sensor causes the drone to hold.

Low or unavailable communications cause the drone to return.

A FAULT state causes return-or-hold behavior.

A LOST state is never treated as safe.

### Contract reason

`DroneState.sensors_ok` is a dictionary keyed by subsystem/sensor name:

`sensors_ok: dict[str, bool]`

Therefore autonomy checks the reported sensor values rather than treating
the entire field as a Boolean.

### Safety rationale

Continuing a wildfire mission with known sensor failure could cause the drone
to navigate or release water using invalid state information.

Communication loss is treated more conservatively because continued autonomy
may leave the drone unable to coordinate its route or receive updated mission
authority.

### Test evidence

`test_health_manager.py` verifies:

- healthy drone -> CONTINUE
- sensor failure -> HOLD
- bad link -> RETURN
- FAULT -> RETURN_OR_HOLD
- LOST -> unsafe
- invalid link threshold -> rejected

---

## 3. Path Claims

### Decision

A path claim contains:

- waypoint sequence
- valid start time
- valid end time
- optional altitude band

Claims are published to the shared swarm path-claim topic.

### Contract reason

`PathClaim.altitude_band_m` is defined as:

`tuple[float, float] | None`

Autonomy therefore does not reinterpret a scalar altitude value.

### Safety rationale

The path claim communicates the spatial and temporal corridor a drone intends
to use so other autonomy instances can perform deconfliction.

A claim is not itself collision avoidance. It is shared coordination data that
supports collision avoidance.

### Validation

Rejected:

- empty waypoint list
- non-positive time window
- invalid altitude band

### Sequence

Each manager instance increments its local message sequence for generated
claims.

---

## 4. Resource Reserve

Resource decisions are delegated to the existing resource manager.

Autonomy does not invent physical consumption constants when the platform
model already provides them.

The controller requests return-to-launch when the resource assessment says
the mission cannot be completed while maintaining required reserves.

### Principle

The reserve is a safety boundary, not an optimization target.

A drone should not spend its reserve simply because doing so would complete
more mission work.

---

## 5. Water-Drop Safety

Water release is deliberately separated from navigation.

Before release, autonomy verifies the drop conditions including:

- correct drone type
- correct task type
- valid task assignment
- current epoch
- correct hotspot assignment
- valid hotspot state
- confidence threshold
- distance
- altitude
- speed
- available water
- requested release volume

If any safety gate fails, the result is HOLD or rejection rather than release.

### Stale epoch protection

A task belonging to an older swarm epoch must not be executed by a drone
operating under a newer epoch.

This prevents an obsolete assignment from becoming executable after leadership
or swarm authority changes.

---

## 6. Ownership Boundaries

### Autonomy owns

- individual-drone route planning
- coverage waypoint generation
- resource feasibility
- path conflict checking
- local health response
- drop safety gating
- path-claim creation
- individual mission execution decisions

### Swarm brain owns

- task allocation
- prioritization
- leader election
- global assignment authority
- swarm-level failure handling

### Perception owns

- fire detection
- hotspot creation/update
- georeferencing
- tracking
- map/perception data

### Backend/simulator owns

- authoritative state mutation
- simulation physics
- persistent state
- station state
- execution of physical effects

Autonomy should request actions rather than silently mutating authoritative
shared state.

---

## 7. Contract Discipline

Shared contracts are treated as authoritative.

When a local autonomy API conflicts with a shared contract, autonomy is adapted
to the contract instead of changing the contract merely to make local tests
pass.

Examples:

- `DroneState.sensors_ok` is a dictionary.
- `PathClaim.altitude_band_m` is a `(lower, upper)` tuple.
- `RefillGrant` does not contain a refill volume.
- `PowerSystem` does not expose arbitrary fuel mutation.

---

## 8. Integration Tradeoffs

The autonomy layer favors explicit decision objects and injected interfaces
over direct simulator/backend mutation.

This allows:

- deterministic unit tests
- simulation with fake autopilot/payload systems
- separation of decision logic from physical execution
- safer integration with the swarm runner

The simulator/backend remains responsible for applying physical effects.

---

## 9. Test Strategy

Safety behavior is tested at multiple levels:

### Unit tests

Verify individual safety decisions such as:

- invalid paths
- path conflicts
- resource insufficiency
- water-drop rejection
- stale epochs
- wrong hotspot ownership
- sensor failure
- communication failure

### Integration tests

Verify that multiple autonomy components compose correctly:

- health gates mission execution
- path claims are published
- validated water sources support refill decisions
- mission execution respects drop gates

### Full suite

The complete autonomy test suite is the regression gate before integration.

---

## 10. Current Phase

Phase 2 is integrating:

- health manager
- path claim manager
- refill manager
- drone controller
- phase-level safety tests

No milestone should be committed until targeted tests and the full suite
pass.


# Phase 2 Integration — Safety Decisions

## Health Manager

The health manager is evaluated before mission execution so a drone with
unsafe health state does not proceed with mission actions.

Safety behavior:
- LOST drones are not considered safe for mission execution.
- FAULT drones require return/hold behavior.
- Failed sensors are treated as a safety condition rather than ignored.
- Link quality is checked against the configured minimum threshold.
- The manager returns an explicit decision instead of allowing a lower-level
  mission objective to override the health state.

## Path Claims

Path claims are represented using the shared `PathClaim` contract.

Autonomy does not redefine the swarm message schema. The path claim manager
adapts drone navigation data to the authoritative contract and preserves the
validity interval and altitude band supplied by the caller.

This keeps deconfliction data explicit and allows the swarm layer to reason
about which path is reserved and for how long.

## Refill

Refill execution is treated as a resource-safety operation rather than
simply a navigation operation.

The drone must use a validated water source/station before refill execution.
Navigation, arrival, payload filling, and resource state remain separate
steps so that a navigation success cannot automatically be interpreted as a
successful refill.

## Mission Execution

The drone controller now owns the execution state machine for an
extinguishing mission.

Normal flow:

    VALIDATE
        |
        v
    TRANSIT
        |
        v
    ARRIVAL CHECK
        |
        v
    WATER-DROP SAFETY CHECK
        |
        v
    RELEASE
        |
        v
    DONE / FAILED

Important safety decisions:

1. A second mission cannot start while one mission is active.
2. The task must be an extinguishing task.
3. The task must be assigned to the executing drone.
4. The task epoch must match the drone epoch.
5. The task must target the supplied hotspot.
6. A hotspot explicitly assigned to another drone is rejected.
7. An unassigned hotspot (`assigned_to=None`) is allowed when the task itself
   is already assigned to the drone.
8. The autopilot must report arrival before water is released.
9. Water-drop safety gates remain authoritative at release time.
10. Unsafe drop conditions cause the drone to hold rather than release water.
11. A payload release returning zero volume produces FAILED rather than DONE.
12. A completed mission cannot release water a second time.

The hotspot assignment rule deliberately distinguishes:

    None        -> no conflicting hotspot assignment
    this drone  -> explicitly assigned to this drone
    other drone -> conflicting assignment; reject

This prevents a competing drone assignment from being ignored without
requiring every hotspot to have a separate assignment field populated before
the task allocator can dispatch it.

## Epoch / Stale Command Protection

Task epoch validation prevents a command from an older swarm decision from
being executed after the drone has moved to a newer swarm epoch.

Therefore:

    task.epoch != drone.epoch
        -> reject before navigation

This prevents stale mission commands from reaching physical mission actions.

## Test Coverage

Phase 2 targeted tests cover:

- health failures
- sensor failures
- path claim creation
- refill integration
- extinguishing mission navigation
- cruise-speed propagation
- arrival gating
- water release
- duplicate-release prevention
- unsafe altitude rejection
- failed payload release
- conflicting task assignment
- stale task epoch
- active mission protection

The full autonomy test suite passes after Phase 2 integration.

## Ownership Boundary

The drone controller owns per-drone execution safety and physical action
gating.

The swarm brain remains responsible for higher-level:
- task allocation
- leadership
- swarm-wide prioritization
- communication
- failure handling

The perception layer remains responsible for detecting and reporting
environmental observations.

Autonomy consumes those outputs and applies local safety gates before
executing physical actions.

## Phase 2 Verification

Targeted Phase 2 tests: PASS

Full test suite: PASS

No Phase 2 milestone should be considered complete unless both targeted
autonomy tests and the full repository test suite remain green.

## Phase 3 — Swarm Integration Decisions

### Swarm adapter boundary

`autonomy/swarm_adapter.py` is the boundary between swarm-wide decisions and
one drone's local execution.

The swarm brain owns allocation and swarm-wide decisions. The local adapter
does not reinterpret those decisions as unconditional permission to perform
physical actions.

### Assignment safety

A task assignment is executable only when:

- the task belongs to this drone;
- the assignment epoch matches the drone epoch;
- the task epoch matches the drone epoch;
- the task status is executable;
- the required target hotspot is locally known;
- the target is not explicitly assigned to another drone;
- the local controller accepts the mission.

Stale assignments are rejected before navigation or physical action.

### Health safety

Before advancing an active mission, local health is evaluated.

Unsafe health causes the controller to hold when an autopilot is available and
blocks mission execution. The adapter publishes a warning alert rather than
continuing because a swarm command cannot override a local safety condition.

### Message ownership

Heartbeat and hotspot reports update local information.

Refill grants are retained only when they are addressed to this drone and match
the current epoch.

Path claims are retained only for this drone's local path state.

These messages do not bypass local health, resource, geometry, or mission
safety checks.

### Completion reporting

The local controller produces `MissionExecutionResult` with execution-level
fields: status, success, water used, and reason.

The adapter maps those fields into the shared `TaskComplete` contract.
`result_status` is left as `None` because the local controller does not own a
`HotspotStatus` transition. The adapter does not invent shared-contract state.

### Alert contract

Autonomy safety rejections use the existing shared `Alert` contract with
`Severity.WARNING` and `AlertKind.GENERIC` because the shared enum does not
define a dedicated autonomy alert kind.

### Integration principle

Shared contracts are treated as authoritative. When the autonomy layer needs
different information, the adapter adapts to the contract rather than
changing the contract solely to simplify autonomy implementation.

### Tests proving the boundary

Phase 3 integration tests verify:

- required swarm subscriptions exist;
- heartbeat messages update local state without physical action;
- hotspot reports update the local target;
- assignments for another drone are ignored;
- autonomy alerts satisfy the exact shared `Alert` contract;
- refill messages for another drone do not alter local refill state.

## Phase 4: Failure and Safety Integration

Phase 4 integrates the existing per-drone mission controller with higher-level
health and swarm-safety constraints. The goal is that a mission action cannot
continue merely because the task is valid; the drone must also be healthy and
able to operate safely.

### Health gate

The swarm autonomy adapter evaluates drone health before attempting mission
execution.

The health hierarchy is:

1. LOST: mission execution is blocked.
2. FAULT: mission execution is blocked and the drone enters the conservative
   hold/return handling defined by the adapter.
3. Sensor failure: mission execution is blocked and the drone holds.
4. Degraded communication link: mission execution is blocked and the drone
   returns to launch.
5. Healthy: mission execution may continue to the mission controller.

The degraded-link threshold is provided by the health manager rather than
invented inside the swarm adapter. A link quality below the configured
minimum is therefore treated as unsafe for continued autonomous mission
execution.

### Path-claim integration

The swarm adapter keeps the latest valid path claim from remote drones and
checks proposed paths against those active remote claims.

A remote claim is considered current only while:

    valid_from <= current_time <= valid_until

Expired claims are ignored rather than treated as permanent obstacles.

Conflict detection remains delegated to the existing 3D deconfliction module.
The adapter therefore owns message/state integration while the deconfliction
module owns the geometric conflict calculation.

### Mission validity remains separate from health

After the health gate passes, the adapter still delegates mission validity and
water-drop safety to the existing DroneController.

This preserves the safety hierarchy:

    health / hard safety
        -> mission validity
        -> resource safety
        -> path / deconfliction
        -> mission action

A valid task therefore cannot override an unhealthy drone state.

### Altitude safety input

The adapter supplies the drone's current altitude to the mission controller
as altitude_agl_m when executing an extinguish step. The existing controller
uses this value as part of the final water-drop safety gate.

The current simulation/contract convention represents altitude using the
Position z component. No additional terrain or elevation model is introduced
because that information is not part of the current authoritative autonomy
contract.

### Failure behavior

The adapter deliberately prevents mission execution when safety prerequisites
fail. It does not fabricate a successful mission completion or release water
when the health gate rejects execution.

For degraded communications, the current deterministic behavior is return to
launch. For sensor failure and the conservative FAULT/LOST handling, the
adapter holds the drone unless the health decision explicitly requests
return.

These behaviors are local autonomy decisions and are kept separate from the
swarm-level decision of how the overall mission should be reassigned.

### Task completion

Terminal mission results (DONE, FAILED, or REJECTED) are converted into the
existing TaskComplete swarm message.

The MissionExecutionResult contract does not contain a HotspotStatus result
field, so TaskComplete uses result_status=None rather than inventing a status.

### Test coverage

Phase 4 integration tests verify:

- healthy mission execution reaches the controller;
- degraded communication blocks mission execution;
- failed sensors block mission execution;
- LOST drones cannot continue the mission;
- FAULT drones cannot continue the mission;
- active remote path claims are checked for conflicts;
- expired remote path claims do not create stale conflicts;
- terminal mission results publish TaskComplete;
- the adapter passes altitude safety information into mission execution.

The test suite passes with all existing autonomy and swarm integration tests.

### Ownership boundary

The swarm autonomy adapter owns swarm-message integration, local state
synchronization, health gating, and translation between swarm contracts and
the per-drone controller.

The DroneController owns executable mission sequencing and final water-drop
safety.

The individual safety managers remain responsible for their specific
decisions. Phase 4 therefore integrates the safety layers without replacing
or duplicating their logic.
