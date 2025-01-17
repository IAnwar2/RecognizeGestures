# RecognizeGestures
## Overview
RecognizeGestures is meant as an accessory to the AutoDoorLock project. The goal of this project is to be able to recoginze if a user is showing a thumbs up, thumbs down or none. Works for Python 3.12.x and under.

## Features
- Using a Fully Connect Neural Network to classify into one of three hand states: Thumbs up, thumbs down, or blank
- Uses mediapipe library to collect relative hand landmark coordinates.
- Real-time gesture detection with opencv and tensorflow
- Easy integration with other applications

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/RecognizeGestures.git
    ```
2. Navigate to the project directory:
    ```bash
    cd RecognizeGestures
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the gesture recognition script:
    ```bash
    python TestGestures.py
    ```
2. Follow the on-screen instructions to perform gestures.

## Adding/Alterng Data
- Run CreateGesture.py to create additional data points for training the model

## Training the Model
- Run HandGuesture.ipynb to retrain and save the model with your new parameters

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.