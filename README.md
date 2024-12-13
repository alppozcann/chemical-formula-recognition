# Chemical Formula Recognition

This repository aims to develop a system for detecting and analyzing chemical formulas from images. The project uses deep learning and computer vision techniques to identify and extract chemical structures and related information from scientific diagrams or schematics.

	Note: The project is still under development, and the codebase may be incomplete or subject to significant changes.

## Overview

	The primary objective of this project is to create a robust tool that:
	•	Identifies chemical formulas and structures in images.
	•	Detects specific components like arrows, bonds, and atom symbols.
	•	Provides bounding box coordinates and recognition results for further analysis.

## Features
	•	Image Preprocessing:
	•	Grayscale conversion, edge detection, and other preprocessing steps for enhancing input images.
	•	Deep Learning Model:
	•	A UNet-based model is utilized for detecting specific features such as arrowheads and bonds in images.
	•	Connected Component Analysis:
	•	Extracts bounding boxes and centroids of detected components for structural analysis.
	•	Line Detection:
	•	Hough Transform is used to detect lines, with the ability to match lines with connected components.
	•	Visualization:
	•	The results, including detected components and lines, are visualized using Matplotlib.

## Current Progress

	The current implementation includes:
	1.	Preprocessing images using OpenCV.
	2.	Detecting arrowheads and connected components using a pre-trained UNet model.
	3.	Applying Hough Transform to detect lines in the images.
	4.	Calculating intersection and overlap between lines and connected components.
	5.	Visualizing results to evaluate the system’s performance.

However, certain aspects such as chemical structure recognition and advanced overlap-based filtering are still under development.
