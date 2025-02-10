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


if (isOpen("Results")) {
    run("Clear Results");
}

// Check if an image is open
if (nImages == 0) {
    exit("No image is open.Choose a seed file");
}

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


