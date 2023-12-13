# Face Mesh Detection

This is a simple program that detects faces in a video feed using the Dlib library and its pre-trained models. The detected faces are then drawn on the frame using their facial landmarks.

## Prerequisites

To run this program, you will need:

- Python 3.x
- OpenCV
- Dlib
- NumPy

## Installation

Clone the repository and install the required libraries using pip:

  https://github.com/wajdisawa/face_detection

Make sure you have the pre-trained models downloaded and saved in the same directory as the script. The models are:

- `shape_predictor_81_face_landmarks.dat`
- `shape_predictor_68_face_landmarks.dat`

You can download these models from the [Dlib website](http://dlib.net/).

## Usage

To run the program, execute the following command:

  python detect_faces.py


This will open a window displaying the video feed with the detected faces and their corresponding facial landmarks. The program will terminate if the 'q' key is pressed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.