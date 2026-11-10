"""A Labels layer that previews a mask on the slice the viewer is showing.

Thresholding one slice takes milliseconds, so it can follow a slider live,
whereas a whole volume (up to ~1e9 voxels) cannot. The layer's buffer is
allocated once and only the displayed slice is rewritten, so dragging does not
allocate a new volume each time and keeps the user's colour/opacity settings.
"""
import numpy as np


class SlicePreview:
    def __init__(self, viewer, name, color=None, opacity=0.6):
        """
        Parameters
        ----------
        viewer : napari.Viewer
        name : str, layer name
        color : RGBA tuple for label 1, or None for napari's default label colours
        """
        self.viewer = viewer
        self.name = name
        self.color = color
        self.opacity = opacity
        self.ref = None          # image layer that defines shape / scale / slicing
        self.layer = None
        self._buf = None
        self._written = None

    # ------------------------------------------------------------------
    def set_reference(self, layer):
        """Image layer the preview follows; changing it removes the preview."""
        if layer is not self.ref:
            self.remove()
            self.ref = layer

    def active(self):
        return self.layer is not None and self.layer in self.viewer.layers

    def current_slice(self):
        """Index tuple of the displayed 2D slice of the reference, or None.

        None when there is no reference, the view is 3D, or the displayed
        slice lies outside the data.
        """
        ref = self.ref
        if ref is None:
            return None
        if ref.ndim == 2:
            return (slice(None), slice(None))
        if self.viewer.dims.ndisplay != 2:
            return None
        offset = self.viewer.dims.ndim - ref.ndim
        axes = [a - offset for a in self.viewer.dims.not_displayed if a >= offset]
        if len(axes) != 1:
            return None
        ax = axes[0]
        coord = ref.world_to_data(self.viewer.dims.point)
        idx = int(round(coord[ax]))
        if not 0 <= idx < ref.data.shape[ax]:
            return None
        sl = [slice(None)] * ref.ndim
        sl[ax] = idx
        return tuple(sl)

    def show_slice(self, mask_fn):
        """Draw mask_fn(slice_index) on the displayed slice only.

        Returns the 2D mask that was drawn, or None if no 2D slice is shown.
        """
        sl = self.current_slice()
        if sl is None:
            return None
        m = mask_fn(sl)
        self._ensure_buffer()
        if self._written is not None:
            self._buf[self._written] = 0
        self._buf[sl] = m
        self._written = sl
        self.layer.refresh()
        return m

    def remove(self):
        lay, self.layer = self.layer, None
        self._buf = None
        self._written = None
        if lay is not None and lay in self.viewer.layers:
            self.viewer.layers.remove(lay)

    # ------------------------------------------------------------------
    def _ensure_buffer(self):
        shape = self.ref.data.shape
        if self._buf is None or self._buf.shape != shape or not self.active():
            self._buf = np.zeros(shape, np.uint8)
            self._written = None
            self._ensure_layer(self._buf)

    def _ensure_layer(self, data):
        if self.active():
            self.layer.data = data
            return
        kwargs = dict(name=self.name, opacity=self.opacity,
                      scale=self.ref.scale, translate=self.ref.translate)
        if self.color is not None:
            try:
                from napari.utils.colormaps import DirectLabelColormap
                kwargs["colormap"] = DirectLabelColormap(color_dict={None: (0, 0, 0, 0), 1: self.color})
            except Exception:
                pass
        active = self.viewer.layers.selection.active
        self.layer = self.viewer.add_labels(data, **kwargs)
        if active is not None:   # don't steal the user's layer selection
            self.viewer.layers.selection.active = active
