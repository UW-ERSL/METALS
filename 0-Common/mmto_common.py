"""MMTO-specific extension of TOParams.

Keeps topopt_common (maintained upstream) free of multi-material / AL fields.
Only the fields the MMTO pipeline reads *beyond* the base TOParams live here:
  - MaterialFilterRadius : separate filter radius for the latent/material block
  - current_major_iter   : major-iterate counter the AL update cadence keys off
  - AL_state             : dict holding the augmented-Lagrangian state (mu, lambda)

The three MMTO TO_QOI members (MASS_AL_LOCAL_STRESS, DUMMY_ZERO, GRAYNESS) are
NOT added here -- Python forbids extending an enum that already has members, so
those are appended to TO_QOI in topopt_common upstream.
"""
from PyTOImports import TOParams  # github base TOParams, via the path-setup layer
from dataclasses import dataclass


@dataclass(slots=True)
class MMTOParams(TOParams):
    MaterialFilterRadius: float | None = None
    current_major_iter: int = -1
    AL_state: dict | None = None
