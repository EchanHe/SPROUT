"""Check the boundary mask when Generate Seeds / Start Growth is clicked.

A mask that cannot be used gives a red line under the Boundary combo (no dialog) and the
run does not start.

SPROUT needs a boundary with exactly two values (0 = keep, non-zero = boundary).
Without this check a wrong layer only fails inside the background worker, with a
less specific message.
"""
import numpy as np


def boundary_problem(data):
    """None if `data` works as a SPROUT boundary, else a short description.

    Mirrors config_core.check_and_cast_boundary: bool, or exactly two values one of
    which is 0. ~1 s on 1e9 voxels (np.unique would take ~20 s).
    """
    data = np.asarray(data)
    if data.dtype == bool:
        return None if data.any() else "it is empty (no boundary voxels)"
    mx, mn = data.max(), data.min()
    if mx == 0 and mn == 0:
        return "it is empty (no boundary voxels)"
    if mn != 0 and mx != 0 and not np.any(data == 0):
        return "it has no 0 voxels (everything would be boundary)"
    if np.any((data != 0) & (data != mx)):
        step = max(1, int(round((data.size / 2e6) ** (1 / data.ndim))))
        values = np.unique(data[(slice(None, None, step),) * data.ndim])
        shown = ", ".join(str(v) for v in values[:6]) + (", ..." if values.size > 6 else "")
        return f"it has more than two values ({shown})"
    return None


def boundary_message(name, problem):
    return (f"The boundary mask '{name}' cannot be used: {problem}.\n"
            "A boundary mask needs exactly two values: 0 = keep, non-zero = boundary.\n"
            "Choose another layer, or set Boundary Mask to None.")


def boundary_ok(parent, name, data, note=None):
    """True if the boundary can be used; otherwise say why in `note` (the red line under
    the Boundary combo, a utils.status.StatusLine) and return False. No dialog."""
    if note is not None:
        note.clear()
    if data is None:
        return True
    problem = boundary_problem(data)
    if problem is None:
        return True
    if note is not None:
        note.set(boundary_message(name, problem), "error")
    return False
