#!/usr/bin/env python3
"""
Finger Count Detection using OpenCV and MediaPipe
Author: Shudhanshu mishra
Description: Real-time finger counting using computer vision
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import List, Tuple, Optional

class FingerCounter:
    def __init__(self):
        """Initialize the finger counter with MediaPipe hands solution."""
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger tip and PIP landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        self.pip_ids = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        # Colors for visualization
        self.colors = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
        # For FPS calculation
        self.prev_time = 0
        
    def count_fingers(self, landmarks: List) -> int:
        """
        Count the number of extended fingers based on hand landmarks.
        
        Args:
            landmarks: List of hand landmarks from MediaPipe
            
        Returns:
            int: Number of fingers extended (0-5)
        """
        fingers_up = []
        
        # Thumb - Check if tip is to the right of IP joint (for right hand)
        # This is a simplified check and works best when palm faces camera
        if landmarks[self.tip_ids[0]].x > landmarks[self.tip_ids[0] - 1].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Other four fingers - Check if tip is above PIP joint
        for i in range(1, 5):
            if landmarks[self.tip_ids[i]].y < landmarks[self.pip_ids[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
                
        return sum(fingers_up)
    
    def get_hand_label(self, landmarks: List, img_width: int) -> str:
        """
        Determine if hand is left or right based on thumb position.
        
        Args:
            landmarks: Hand landmarks
            img_width: Width of the image
            
        Returns:
            str: 'Left' or 'Right'
        """
        # Simple heuristic: check thumb position relative to wrist
        wrist_x = landmarks[0].x
        thumb_x = landmarks[4].x
        
        if thumb_x > wrist_x:
            return "Right"
        else:
            return "Left"
    
    def draw_finger_count(self, img: np.ndarray, count: int, position: Tuple[int, int], 
                         hand_label: str) -> None:
        """
        Draw finger count on the image.
        
        Args:
            img: Input image
            count: Number of fingers
            position: Position to draw the count
            hand_label: Label for the hand (Left/Right)
        """
        # Draw background rectangle
        cv2.rectangle(img, (position[0] - 20, position[1] - 40), 
                     (position[0] + 100, position[1] + 10), 
                     self.colors['black'], -1)
        
        # Draw finger count
        cv2.putText(img, f"{hand_label}: {count}", position, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)
    
    def draw_landmarks_and_connections(self, img: np.ndarray, landmarks) -> None:
        """
        Draw hand landmarks and connections on the image.
        
        Args:
            img: Input image
            landmarks: Hand landmarks from MediaPipe
        """
        # Draw landmarks
        for idx, landmark in enumerate(landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Color code different parts
            if idx in self.tip_ids:
                cv2.circle(img, (cx, cy), 8, self.colors['red'], -1)
            elif idx in self.pip_ids:
                cv2.circle(img, (cx, cy), 6, self.colors['blue'], -1)
            else:
                cv2.circle(img, (cx, cy), 4, self.colors['green'], -1)
        
        # Draw connections
        self.mp_draw.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def calculate_fps(self) -> int:
        """Calculate and return current FPS."""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return int(fps)
    
    def run(self):
        """Main function to run the finger counter application."""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Finger Counter Started!")
        print("Instructions:")
        print("- Hold your hand(s) in front of the camera")
        print("- Make sure your palm faces the camera")
        print("- Press 'q' to quit")
        print("- Press 's' to save screenshot")
        
        total_finger_count = 0
        
        while True:
            ret, img = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip image horizontally for mirror effect
            img = cv2.flip(img, 1)
            h, w, c = img.shape
            
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_img)
            
            total_finger_count = 0
            hand_info = []
            
            # Process detected hands
            if results.multi_hand_landmarks:
                for idx, (landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):
                    
                    # Draw landmarks and connections
                    self.draw_landmarks_and_connections(img, landmarks)
                    
                    # Count fingers
                    finger_count = self.count_fingers(landmarks.landmark)
                    total_finger_count += finger_count
                    
                    # Get hand label
                    hand_label = handedness.classification[0].label
                    
                    # Store hand info
                    hand_info.append((hand_label, finger_count))
                    
                    # Draw finger count for each hand
                    y_offset = 50 + (idx * 40)
                    self.draw_finger_count(img, finger_count, (50, y_offset), hand_label)
            
            # Draw total finger count
            cv2.rectangle(img, (w - 200, 20), (w - 20, 80), self.colors['black'], -1)
            cv2.putText(img, f"Total: {total_finger_count}", (w - 180, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['yellow'], 3)
            
            # Draw FPS
            fps = self.calculate_fps()
            cv2.putText(img, f"FPS: {fps}", (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['green'], 2)
            
            # Draw instructions
            cv2.putText(img, "Press 'q' to quit, 's' to save", (20, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
            
            # Show the image
            cv2.imshow("Finger Counter", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"finger_count_{timestamp}.jpg"
                cv2.imwrite(filename, img)
                print(f"Screenshot saved as {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Finger Counter stopped.")

def main():
    """Main function to run the application."""
    try:
        counter = FingerCounter()
        counter.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install opencv-python mediapipe numpy")

if __name__ == "__main__":
    main()
