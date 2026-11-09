"""Demo simulator / game for the night initial-attack swarm (see exclude/simulator_task.md).

Python side is authoritative: maps, fire physics, thermal rendering, perception, autonomy and the swarm-brain stubs
all run here. The browser client (frontend/) only draws snapshots and sends player input.

Uses the real `perception/` pipeline and `autonomy/` execution code. Only the swarm brain (zone allocation,
task assignment, refill assignment, leader) is stubbed, in `sim/brain_stub.py`.
"""
