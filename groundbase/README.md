# Ground-base review queue

`GroundReviewQueue` subscribes to the reliable `ground/reviews` topic, rejects expired reports, and deduplicates
pending items by their stable `review_id`. Reports contain the tracked hotspot, fused-confidence score, and the
triage band limits used by the seeker.

The queue is currently in-memory. A production ground-base backend should persist pending and resolved items and
expose them to the operator UI. `resolve()` records a local operator disposition; sending that disposition back to a
drone is not implemented by this queue.
