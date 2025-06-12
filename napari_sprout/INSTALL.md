# Installation Guide for napari-SPROUT

## Prerequisites

1. **SPROUT must be installed or accessible**
   - The napari plugin uses the existing SPROUT code without modifying it
   - SPROUT should be in your Python path or in the parent directory

2. **Python environment**
   - Python 3.8 or higher
   - napari installed

## Installation Steps

### Option 1: Development Installation (Recommended)

1. Navigate to the napari_sprout directory:
   ```bash
   cd napari_sprout
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

### Option 2: Direct Installation

```bash
pip install ./napari_sprout
```

### Option 3: Manual Setup

If you prefer not to install, you can run the example directly:

1. Ensure napari is installed:
   ```bash
   pip install napari[all]
   ```

2. Run the example:
   ```bash
   cd napari_sprout
   python example_usage.py
   ```

## Verifying Installation

1. Start napari:
   ```python
   import napari
   viewer = napari.Viewer()
   napari.run()
   ```

2. Check if SPROUT appears in the Plugins menu

## Using with Existing SPROUT Installation

The plugin automatically tries to import SPROUT from:
1. The Python path (if SPROUT is installed)
2. The parent directory (../sprout_core)

Make sure one of these locations contains the SPROUT modules.

## Troubleshooting

### "SPROUT modules not found" error
- Ensure SPROUT is in the Python path
- Check that sprout_core directory exists in the parent folder
- Try adding SPROUT to path manually:
  ```python
  import sys
  sys.path.append('/path/to/SPROUT')
  ```

### Widget not appearing in napari
- Restart napari after installation
- Check the console for error messages
- Verify the plugin is listed in napari's plugin manager

### Import errors
- Ensure all dependencies are installed:
  ```bash
  pip install numpy scikit-image tifffile pandas pyyaml qtpy
  ```
