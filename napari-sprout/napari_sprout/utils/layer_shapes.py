"""Input layers that do not fit the reference layer (the image, or the layer being edited).

A layer of another shape stays in the combo, greyed out as "<name> (shape mismatch)", so it
cannot be picked. The layer name is the item data (the text has the suffix). Only the shape
(metadata) is read, never the voxels. The core checks the shapes again when it runs.
"""
import numpy as np
from qtpy.QtCore import Qt

MISMATCH = " (shape mismatch)"


def layer_shape(layer):
    """Shape of a layer's data; for a multiscale layer the full-resolution level."""
    if layer is None:
        return None
    data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
    return tuple(data.shape)


def placement_differs(layer, ref):
    """True if two layers of one shape are placed differently in napari (scale or translate)."""
    return not (np.allclose(layer.scale, ref.scale) and np.allclose(layer.translate, ref.translate))


def item_layer(combo, i):
    """Layer name of item i: the item data if set, else the text."""
    data = combo.itemData(i)
    return data if isinstance(data, str) else combo.itemText(i)


def find_layer_item(combo, name):
    for i in range(combo.count()):
        if item_layer(combo, i) == name:
            return i
    return -1


def current_layer_name(combo):
    i = combo.currentIndex()
    return item_layer(combo, i) if i >= 0 else ""


def grey_out_mismatches(combo, viewer, ref):
    """Grey out the layers of `combo` whose shape differs from `ref` (no reference: none greyed).

    If the current item gets greyed, the combo goes to its "None" item, or to no item at all
    (never to another layer the user did not pick). Returns the name of the layer that was
    dropped that way, else None.
    """
    ref_shape = layer_shape(ref)
    model = combo.model()
    current = combo.currentIndex()
    blocked = combo.blockSignals(True)
    try:
        for i in range(combo.count()):
            name = item_layer(combo, i)
            if name not in viewer.layers:
                continue                                  # "None" and other non-layer items
            bad = ref_shape is not None and layer_shape(viewer.layers[name]) != ref_shape
            item = model.item(i)
            if item is not None:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled if bad else item.flags() | Qt.ItemIsEnabled)
            combo.setItemData(i, name)
            text = name + MISMATCH if bad else name
            if combo.itemText(i) != text:
                combo.setItemText(i, text)
    finally:
        combo.blockSignals(blocked)
    if current < 0 or _enabled(combo, current):
        return None
    dropped = item_layer(combo, current)
    no_layer = [i for i in range(combo.count()) if item_layer(combo, i) not in viewer.layers]
    combo.setCurrentIndex(no_layer[0] if no_layer else -1)    # signals (unless blocked): the panel follows
    return dropped


def _enabled(combo, i):
    item = combo.model().item(i)
    return item is None or bool(item.flags() & Qt.ItemIsEnabled)
