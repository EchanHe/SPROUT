"""Layers and the files they came from, for the YAML files of the tabs."""
import os


def source_path(layer):
    p = getattr(getattr(layer, "source", None), "path", None) if layer is not None else None
    return str(p) if p else None


def same_path(a, b):
    try:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))
    except Exception:
        return False


def yaml_path(viewer, name, unsaved):
    """The file of layer `name` for a YAML. A layer without a file keeps its name and goes in `unsaved`."""
    if name not in viewer.layers:
        return name
    path = source_path(viewer.layers[name])
    if path:
        return path
    unsaved.append(name)
    return name


def layer_name(viewer, value):
    """The layer a YAML value means: the layer opened from that file, else `value` as a layer name."""
    layer = next((l for l in viewer.layers if source_path(l) and same_path(source_path(l), value)), None)
    return layer.name if layer is not None else value


def saved_note(file_name, unsaved):
    """Status text and level after an export."""
    if not unsaved:
        return f"Parameters successfully saved to {file_name}.", "ok"
    return (f"Saved {file_name}. No file for {', '.join(unsaved)}: the YAML holds the layer name, "
            "set the path before using it from the command line."), "warn"
