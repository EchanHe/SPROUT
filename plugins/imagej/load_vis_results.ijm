// Ask user to select a folder containing TIFF images
folder = getDirectory("Choose a folder containing TIFF images");

// Get list of all TIFF files in the folder
list = getFileList(folder);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".tiff") || endsWith(list[i], ".tif")) {
        // Open the image
        open(folder + list[i]);
        
        // Apply Glasbey LUT
        run("glasbey");
        
        // Update the display (optional)
        wait(500); // Wait briefly to avoid flickering when showing images
    }
}
