# Decision records

Short records of design decisions: context, options, what we chose and why. Written to be presented in the demo
("Judge question / Our answer" at the top of each).

| # | Decision | Status |
|---|---|---|
| [0001](0001-seeker-search-altitude-vs-lidar-range.md) | Seeker searches at 80 m; lidar guards the flight path; add a ≥ 100 m altimeter (alternatives: long-range lidar, downward lidar, radar) | Accepted (A) |
| [0002](0002-perception-sensor-wiring-and-lens-distortion.md) | Perception sensor wiring (pose sync, terrain/AGL, RTK loss, camera health) and lens distortion | Accepted |
| [0003](0003-hot-blob-bypass.md) | Hot-blob bypass: the ML can't veto an obvious fire (replaces fixed 300 °C) | Accepted |
| [0004](0004-close-range-confirmation.md) | Close-range confirmation: same CNN, scale-normalized patches, separate precision-first threshold, multi-frame | Accepted |
