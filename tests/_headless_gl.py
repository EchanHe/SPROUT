"""Test helper: napari >= 0.7 under QT_QPA_PLATFORM=offscreen (no OpenGL context).

napari 0.7 queries the GL driver when a Viewer / an image layer is made
(get_max_texture_sizes: PyOpenGL raises GLError without a context; the tiled image node
then gets tile_size=None) and Viewer() or add_image() fail. napari 0.6.1 never needed this.

`import _headless_gl` before napari.Viewer(...) in a check: with QT_QPA_PLATFORM=offscreen
and napari >= 0.7, the two GL queries return fixed values (the same names are patched in
the napari._vispy modules that imported them). Otherwise (0.6.x, or a real display) it does
nothing. Test harness only.
"""
import importlib
import os

MAX_TEXTURE_SIZES = (16384, 2048)       # (2D, 3D), what a typical desktop GPU reports
GL_EXTENSIONS = "GL_ARB_texture_float"
MODULES = ("napari._vispy.utils.gl", "napari._vispy.canvas", "napari._vispy.layers.base",
           "napari._vispy.layers.image", "napari._vispy.layers.labels")
PATCHED = False


def _sizes():
    return MAX_TEXTURE_SIZES


def _extensions():
    return GL_EXTENSIONS


def _napari_version():
    import napari
    try:
        return tuple(int(p) for p in napari.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


def install():
    """Patch the GL queries (offscreen + napari >= 0.7 only). Returns True if patched."""
    global PATCHED
    if PATCHED:
        return True
    if os.environ.get("QT_QPA_PLATFORM") != "offscreen" or _napari_version() < (0, 7):
        return False
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        for attr, fn in (("get_max_texture_sizes", _sizes), ("get_gl_extensions", _extensions)):
            if hasattr(mod, attr):
                setattr(mod, attr, fn)
    PATCHED = True
    return True


install()
