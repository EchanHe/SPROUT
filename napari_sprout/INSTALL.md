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