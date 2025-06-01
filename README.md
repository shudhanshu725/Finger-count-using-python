# Finger-count-using-python

# Finger Count Detection

A real-time finger counting application using computer vision that can detect and count fingers from one or both hands using your webcam.

![Finger Count Detection Demo](demo.gif)

## Features

- Real-time finger counting using webcam feed
- Support for both single and dual hand detection
- Visual feedback with hand landmarks and connections
- FPS counter
- Screenshot capability
- Hand labeling (Left/Right)
- Total finger count display
- Mirror effect for intuitive interaction

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/finger-count-detection.git
cd finger-count-detection
```

2. Install the required packages:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

1. Run the application:
```bash
python fig.py
```

2. Hold your hand(s) in front of the camera
3. Make sure your palm faces the camera for best results
4. The application will display:
   - Hand landmarks and connections
   - Finger count for each hand
   - Total finger count
   - Current FPS

### Controls

- Press 'q' to quit the application
- Press 's' to save a screenshot

## How It Works

The application uses MediaPipe's hand tracking solution to detect hand landmarks in real-time. It then analyzes the relative positions of finger joints to determine which fingers are extended. The system can handle both left and right hands simultaneously.

Key features:
- Hand landmark detection using MediaPipe
- Finger state analysis based on joint positions
- Real-time visualization of hand tracking
- Support for multiple hands
- Automatic hand labeling (Left/Right)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
