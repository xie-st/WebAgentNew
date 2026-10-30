"""Backward-compatibility test suite.

Guards the public surface and serialized-data formats that v0.2.x users depend on.
``public_surface_v0_2.json`` is the golden snapshot; the live surface must remain a
superset of it (additions are fine; removals/renames must be intentional and update
the snapshot deliberately).
"""
