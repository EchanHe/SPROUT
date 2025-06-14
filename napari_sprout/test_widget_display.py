"""Test script to check if the SPROUT widget displays correctly."""

import napari
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Test the widget display."""
    print("Testing SPROUT widget display...")
    
    # Create viewer
    viewer = napari.Viewer()
    
    # Load demo image
    try:
        import tifffile
        demo_path = Path(__file__).parent.parent / "data" / "demo_dog.tiff"
        if demo_path.exists():
            image = tifffile.imread(str(demo_path))
            viewer.add_image(image, name="demo_dog")
            print(f"Loaded demo image: {demo_path}")
    except Exception as e:
        print(f"Could not load demo image: {e}")
    
    # Try to create and add the widget directly
    try:
        from napari_sprout._widget import make_sprout_widget
        
        # Create widget
        widget = make_sprout_widget(viewer)
        
        # Check widget properties
        print(f"Widget created: {widget}")
        print(f"Widget type: {type(widget)}")
        print(f"Widget visible: {widget.isVisible()}")
        print(f"Widget size: {widget.size()}")
        
        # Add to dock
        dock_widget = viewer.window.add_dock_widget(
            widget,
            name="SPROUT Test",
            area="right"
        )
        
        # Force show
        widget.show()
        widget.raise_()
        
        print("Widget added to dock successfully!")
        
        # Check if widget has content
        if hasattr(widget, 'tabs'):
            print(f"Widget has tabs: {widget.tabs}")
            print(f"Tab count: {widget.tabs.count()}")
        
    except Exception as e:
        import traceback
        print(f"Error creating widget: {e}")
        traceback.print_exc()
    
    # Start napari
    napari.run()


def test_debug():
    # launch_napari.py
    from napari import Viewer, run

    viewer = Viewer()
    dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
        "napari-sprout", "SPROUT_info"
    )
    dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
        "napari-sprout", 'SPROUT_edit'
    )
    

    # Optional steps to setup your plugin to a state of failure
    # E.g. plugin_widget.parameter_name.value = "some value"
    # E.g. plugin_widget.button.click()
    run()

if __name__ == "__main__":
    # main()
    test_debug()
