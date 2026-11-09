"""Swarm brain (workstream A): pure logic, imports only `contracts` + stdlib (hard rule 1).

  priority    fire threat score from what perception reports (size, growth, temperature, confidence)
  allocation  refill-aware travel cost + the priority-queue task allocator
  zones       seeker search zones (equal-time bands)
  leader      bus-facing leader role that ties them together (replaces sim/brain_stub.StubLeader for the swarm)
"""
