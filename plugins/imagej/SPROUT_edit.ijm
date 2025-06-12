// Function to get unique sorted values
function unique(arr) {
    uniqueList = newArray(); // Store unique values

    // Check if value already exists
    for (i = 0; i < lengthOf(arr); i++) {
        found = false;
        for (j = 0; j < lengthOf(uniqueList); j++) {
            if (arr[i] == uniqueList[j]) {
                found = true;
                break;
            }
        }
        if (!found) {
            uniqueList = Array.concat(uniqueList, arr[i]);
        }
    }

    return Array.sort(uniqueList); // Sort and return unique values
}

// Funciton to split connected componenets into different classes
function splitCcopm(pixel_value, min_area) {
	// Store the current active image (Image1)
	id1 = getImageID();
	title1 = getTitle();

	if(pixel_value!=-1){
		max = pixel_value
	}else{
		getStatistics(area, mean, min, max, std, histogram);
	}

	new_max = max;
	for (pixel_value_temp = 1; pixel_value_temp <= max; pixel_value_temp++) {
		print(new_max);
		print(max);
	
		// Duplicate the original mask
		run("Duplicate...", "title=Connected_Components");
		
		// Store the current active image (Image1)
		id2 = getImageID();
		title2 = getTitle();
	
	
		print("Processing value: " + pixel_value_temp);
	
	    // Create a mask for the current pixel value
	    setThreshold(pixel_value_temp, pixel_value_temp);
	    run("Convert to Mask");
	
	    // Analyze connected components for this value
	    run("Analyze Particles...", "size=0-Infinity show=Nothing clear add");	
	
			// Get the number of detected components
	    num_components = roiManager("count");
	    print("  -> Found " + num_components + " components for value " + pixel_value_temp);
		if (num_components >1){
			selectImage(id1);
		    // Assign unique labels (value + component_id)
		    for (j = 1; j < num_components; j++) {
		        roiManager("Select", j);
				getStatistics(area);
		        if (area < min_area) {
		            run("Set...", "value=0"); // Set to background
		            print("    -> Small component (Area: " + area + ") removed.");
		        } else {
		            run("Set...", "value=" + (new_max)); // Assign a new unique label
		            new_max = new_max + 1;	      
		        }
		    }
		
	
		}
	    roiManager("Deselect");
	    roiManager("Reset");
	    selectImage(id2);
	    run("Close");
		run("Select None"); 
	}

}


// Funciton to to array into string
function arrayToString(arr) {
    str = "";
    for (i = 0; i < lengthOf(arr); i++) {
    	str_temp = "" + arr[i];
        str += str_temp;
        if (i < lengthOf(arr) - 1) {
            str += ", ";
        }
    }
    return str;
}

// Function to check if an integer is in a list
function isInList(value, arr) {
    for (i = 0; i < lengthOf(arr); i++) {
        if (arr[i] == value) {
            return true; // Found, return true
        }
    }
    return false; // Not found, return false
}

// end of functions 


if (isOpen("Results")) {
    run("Clear Results");
}

// Check if an image is open
if (nImages == 0) {
    exit("No image is open.Choose a seed file");
}


if (selectionType == 9 || selectionType == 10){

	// Create a pop-up dialog
	Dialog.create("Choose an Option to set pixel values");
	Dialog.addMessage("Working on image: "+getTitle())
	
	Dialog.addRadioButtonGroup("Set Value To:", newArray("Smallest Value", "Zero", "Custom Number", "Selected Only"), 3, 1, "Smallest Value");
	//Dialog.addNumericField("Enter Custom Value:", 0, 0); // Custom number input
	Dialog.addNumber("Enter Custom Value:", 512);
	//Dialog.addString("Title:","asd");
	Dialog.show();
	
	// Get user selection
	choice = Dialog.getRadioButton();
	customValue = Dialog.getNumber();
	
	
	
	// Check if the image has only one channel
	//channels = getDimensions()[2];
	//if (channels > 1) {
	//    exit("This macro only works with single-channel images.");
	//}
	
	// Ask the user to select points
	//print("Click on several points, then press OK.");
	roiManager("reset");
	//waitForUser("Click on multiple points, then click OK.");
	
	// Get selected coordinates
	roiManager("Add");
	roiManager("Measure");
	//numPoints = roiManager("count");
	numPoints = nResults;
	if (numPoints == 0) {
	    exit("No points selected.");
	}
	
	
	
	
	// Get the pixel values at the selected points
	selectedValues = newArray(numPoints);
	//for (i = 0; i < numPoints; i++) {
	//    roiManager("Select", i);
	//    run("Measure");
	//    selectedValues[i] = getResult("Mean", i); // Get pixel value
	//}
	
	// Print all measured values
	for (i = 0; i < numPoints; i++) {
	    value = getResult("Mean", i); // Get pixel value from results table
	    print("ROI " + (i+1) + " Pixel Value: " + value);
	    selectedValues[i] = value;
	}
	// Sort the array in ascending order
	selectedValues = Array.sort(selectedValues);
	selectedValues = unique(selectedValues);
	//print("Value selected complete", selectedValues);
	print("Value selected complete " + arrayToString(selectedValues));
	
	
	newValue = 0;
	// Choices for setting the newValue
	if (choice == "Smallest Value") {
	    newValue = selectedValues[0]; // Set to the smallest value
	    print("Option Chosen: Set to Smallest Value (" + newValue + ")");
	} else if (choice == "Zero") {
	    newValue = 0; // Set to zero
	    print("Option Chosen: Set to Zero");
	}else if (choice == "Custom Number") {
	    newValue = customValue; // Set to custom value
	    print("Option Chosen: Set to Custom Value (" + newValue + ")");
	}
	// Ask for new pixel value
	//newValue = getNumber("Enter new pixel value:", 10);
	
	
	// Create a Yes/No dialog of printing the values to process, whether to countinue
	Dialog.create("Confirmation");
	Dialog.addMessage("Do you want to continue?");
	Dialog.addMessage("Working on image: : "+getTitle())
	if (choice == "Selected Only"){
	    Dialog.addMessage("	Keep regions only from: " + arrayToString(selectedValues));
	}
	else{
	    Dialog.addMessage("	Old Values: " + arrayToString(selectedValues));
	    Dialog.addMessage("	New Values: " + newValue);
	}
	
	Dialog.addChoice("Select an option:", newArray("Yes", "No"), "Yes");
	Dialog.show();
	
	// Get user choice
	choice_step_2 = Dialog.getChoice();
	
	// Check user selection
	if (choice_step_2 == "Yes") {
		// Convert pixels matching selected values to the new value
		width = getWidth();
		height = getHeight();
		
		for (y = 0; y < height; y++) {
		    for (x = 0; x < width; x++) {
		        value = getPixel(x, y);
	
	            if (choice == "Selected Only"){
	                if (!isInList(value, selectedValues)) {
	                    setPixel(x, y, 0);
	                } 
	            }
	            else{
	                for (i = 0; i < lengthOf(selectedValues); i++) {
	                    if (value == selectedValues[i]) {
	                        setPixel(x, y, newValue);
	                        break;
	                    }
	                }
	            }
	
	        showProgress(((x+1)*(y+1)) / (width *height));
	
		    }
		}
		
		
		// Update display
		updateDisplay();
		print("Pixel replacement complete!");
		
		roiManager("reset");
		run("Select None");    
	} 
}else if (selectionType == 0 || selectionType == 1 || selectionType == 2 || selectionType == 3){
		// Create a pop-up dialog
	Dialog.create("Set pixel values for selected regions");
	Dialog.addMessage("Working on image: "+getTitle())
	//Dialog.addNumericField("Enter Custom Value:", 0, 0); // Custom number input
	Dialog.addNumber("Enter Custom Value:", 0);
	//Dialog.addString("Title:","asd");
	Dialog.show();
	
	// Get user selection
	customValue = Dialog.getNumber();
	
	run("Set...", "value=" + customValue);

	roiManager("reset");
	run("Select None"); 
	
	//Dialog.addCheckbox("Perform Split After Processing?", false);  // Checkbox with default "false"
	// Create a pop-up dialog
	Dialog.create("Perform Split After Processing?");
	Dialog.addMessage("Working on image: "+getTitle())
	Dialog.addNumber("Select the class to split, -1 for all classes:", -1);
	Dialog.addNumber("Min area to keep:", 0)
	Dialog.show();
	split_class = Dialog.getNumber();
	min_area = Dialog.getNumber();
	splitCcopm(split_class, min_area);

}




