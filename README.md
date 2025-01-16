# Chemical Formula Recognition

This repository aims to develop a system for detecting and analyzing chemical formulas from images. The project uses deep learning and computer vision techniques to identify and extract chemical structures and related information from scientific diagrams or schematics.

**Note:** This project is under active development, and features may change frequently.

**This project works only with MacOS environment for time being.**

## Overview

The primary objective of this project is to create a robust tool that:

•	Identifies chemical formulas and structures in images.
 
•	Detects specific components like arrows, bonds, and atom symbols.
 
•	Provides bounding box coordinates and recognition results for further analysis.

## Features
- **Image Preprocessing:**  

	• Grayscale conversion, edge detection, and other preprocessing steps for enhancing input images.



- **Deep Learning Model:**

  
	• 	*A UNet-based model is utilized for detecting specific features such as arrowheads and bonds in images.*


 
- **Connected Component Analysis:**
    
	•	*Extracts bounding boxes and centroids of detected components for structural analysis.*

- **Line Detection:**
  
	•	*Hough Transform is used to detect lines, with the ability to match lines with connected components.*

- **Visualization:**
  
	•	*The results, including detected components and lines, are visualized using Matplotlib.*


## Current Progress

The current implementation includes:

1.	Preprocessing images using OpenCV.

2.	Detecting arrowheads and connected components using a pre-trained UNet model.

3.	Applying Hough Transform to detect lines in the images.
   
4.	Calculating intersection and overlap between lines and connected components.
   
5.	Visualizing results to evaluate the system’s performance.


# Installing the prerequisites
For installing the prerequisite libraries , please follow these steps:
1. Make sure that you have the followings installed:
   
	- [Python 3.10](https://www.python.org/) or later
   
	- [pip](https://pip.pypa.io/en/stable/)

2. Create a virtual environnement (Optional):
   
   To manage dependincies, it is recommended to create a virtual environment. For creating one:
	```bash
	python -m venv venv
 	```
   To activate the virtual environment:
	- On Windows:
	```bash
 	venv\Scripts\activate
 	```
  	- On Mac/Linux:
	```bash
 	source venv/bin/activate
 	```
3. Install Dependencies:

   Use the requirements.txt file to install all the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify Installation:
   
   You can verify that the installation was successful by running the following command:
	```bash
	python -m pip freeze
	```
