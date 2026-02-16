### Video Understanding

This repo contains the implementation of Video understanding, it primarily uses:

- YOLO for Object Detection
- SlowFast by FAIS for Action detection
- VideoLlama 3 for Video Analysis

The YOLO first detects the people in the video, then passes it on the SlowFast model for Actions detection,
After which the contextualization happens using YOLO's and SlowFast model's output and passed it to 
VideoLlama3 with the video for analysis
