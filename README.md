# ChromaKey Image Processing
This Python script, ChromaKey.py, provides tools for processing images using chroma key techniques, 
including finding the centroid of an object in an image and resizing images based on specific constraints. 
The script is designed to work with image files, leveraging OpenCV for image processing tasks.

## Features
* Color Space Transformation
Read an Image: The script can read an image and display the original color image alongside its components in a specified color space (CIE-XYZ, CIE-Lab, YCrCb, or HSB).
Display Components: The color components are displayed in grayscale, arranged alongside the original image in a single viewing window.

* Chroma Keying
Green Screen Extraction: The script extracts a person from a green screen photo using chroma key techniques.
Background Replacement: It replaces the green screen with a scenic background, aligning the person horizontally to the center and vertically to the bottom of the scenic image.
Display Results: The script displays the original green screen photo, the extracted person, the scenic background, and the combined result in a single viewing window.

## Requirements
* Python 3.x
* OpenCV (cv2)
* NumPy

## Installation
To install the required libraries, you can use pip:
```
pip install opencv-python-headless numpy
```

## Usage
The script is designed to be run from the command line. It supports two tasks, each requiring specific command-line arguments.

Color Space Transformation
```
python ChromaKey.py -XYZ|-Lab|-YCrCb|-HSB imagefile
```

Chroma Keying
```
python ChromaKey.py scenicImageFile greenScreenImageFile
```

Example Commands
Color Space Transformation
```
python ChromaKey.py -Lab input_image.jpg
```

Chroma Keying
```
python ChromaKey.py scenic.jpg green_screen.jpg
```
