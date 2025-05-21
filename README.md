
# ASLens - Sign Language to Text Conversion

ASLens is a deep learning project that convertes sign language gestures into text, using deep recurrent neural networks.

  

# Overview

**ASLens** is an assistive tool designed to promote the inclusion of people with hearing loss by translating sign language gestures into written text. This project leverages deep learning techniques to convert sign language video data into meaningful textual output.

The primary goal of ASLens is to translate sign language gestures into readable text using Recurrent Neural Networks (RNNs)

-   **Dataset**: The system is trained on the How2Sign dataset, which provides a rich collection of sign language video data.
    
-   **Feature Extraction**: Visual features are extracted from the videos using [MediaPipe](https://mediapipe.dev/), which detects and tracks key points of hand and body movements.
    
-   **Sequence Modeling**:
    
    -   An **RNN Encoder** processes the extracted gestures to understand how the signs change and move over time.
        
    -   A **CharRNN Decoder** then translates these encoded features into coherent text sentences, character by character.

## Dataset

### Step 1: Collecting Data

ASLens uses the **[How2Sign ](https://how2sign.github.io/)** dataset, which consists of video data of sign langauge gestures (ASL language), with the corresponding text translation.

### Step 2: Extracting Hand - Face Keypoints using MediaPipe

To extract meaningful features for model training, we use **MediaPipe** to extract hand - face keypoints from the video frames. **[MediaPipe ](https://ai.google.dev/edge/mediapipe/solutions/guide)** provides pre-built solutions for detecting hand - face keypoints, which interpret sign language gestures. Due to limited computational resources, a video frame rate of 15fps was used in this project.




![](https://github.com/farahpiralic/ASLens-Test/blob/main/Wow-gif.gif)
