"""Lock the Labels layers a background operation reads, while it runs.

The Edit and Label Mapping workers read a layer's own array (no copy, to save memory). The
napari brush writes into that array in place, so strokes made during the run would partly
end up in the result and in the Undo state. While the run lasts those layers are not
editable (napari greys out the paint tools); afterwards each gets back its own state.

Reference-counted across docks, so overlapping runs on one layer restore it once, at the
end of the last run. napari resets `editable` when `layer.data` is set, so release after
the result is written. napari also clears the layer's brush history (Ctrl+Z) when it is
locked.
"""
from napari.layers import Labels

_held = {}   # id(layer) -> [layer, runs holding it, editable before the first]


def lock_layers(viewer, arrays):
    """Lock every Labels layer of `viewer` whose data is one of `arrays` (by identity).
    Returns the locked layers (pass them to unlock_layers)."""
    layers = [lay for lay in viewer.layers
              if isinstance(lay, Labels) and any(lay.data is a for a in arrays)]
    for lay in layers:
        entry = _held.get(id(lay))
        if entry is None:
            _held[id(lay)] = [lay, 1, lay.editable]
        else:
            entry[1] += 1
        try:
            lay.editable = False
        except Exception:   # a napari without the property: run unlocked
            pass
    return layers


def unlock_layers(layers):
    """End one run's lock: the last run gives each layer its editable state back."""
    for lay in layers:
        entry = _held.get(id(lay))
        if entry is None:
            continue
        entry[1] -= 1
        if entry[1] == 0:
            del _held[id(lay)]
            try:
                lay.editable = entry[2]
            except Exception:
                pass


def lock_note(layers):
    """The status line shown while `layers` are locked ('' if none)."""
    if not layers:
        return ""
    names = ", ".join(lay.name for lay in layers)
    return f"Running… {names} {'is' if len(layers) == 1 else 'are'} locked until it finishes."
