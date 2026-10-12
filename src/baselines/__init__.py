"""Established real-patch controls used to assess GB-center representation."""

from .farthest_first import global_farthest_first_order
from .medoid import MEDOID_BATCH, medoid_indices

__all__ = ["global_farthest_first_order", "medoid_indices", "MEDOID_BATCH"]
