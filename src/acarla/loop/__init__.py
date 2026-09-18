"""Slow-motion closed loop: CARLA (local) <-> Alpamayo worker (Kaggle).

The simulation is synchronous and simply does not tick while a plan is in
flight, so wall-clock latency of the remote worker never becomes plan age.
`codec` defines the wire format, `transport` the queue on a private Hugging
Face dataset repository; both sides import only these two modules.
"""
