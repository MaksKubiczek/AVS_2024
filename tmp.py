import numpy as np
import cv2

# Function to initialize buffer and counter
def initialize_buffer(height, width, N):
    BUF = np.zeros((height, width, N), dtype=np.uint8)
    iN = 0
    return BUF, iN

# Function for mean calculation
def compute_mean(BUF):
    mean_frame = np.mean(BUF, axis=2).astype(np.uint8)
    return mean_frame

# Function for median calculation
def compute_median(BUF):
    median_frame = np.median(BUF, axis=2).astype(np.uint8)
    return median_frame

# Function for background subtraction and binarization
def background_subtraction(current_frame, model_frame):
    diff_frame = cv2.absdiff(current_frame, model_frame)
    _, binary_diff = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
    return binary_diff

# Main function to process frames
def process_frames(frames, N):
    height, width, _ = frames[0].shape
    BUF, iN = initialize_buffer(height, width, N)

    # Process frames
    for frame in frames:
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Store current frame in buffer
        BUF[:, :, iN] = gray_frame

        # Increment counter
        iN += 1
        if iN >= N:
            iN = 0
        
        # Compute mean and median
        mean_frame = compute_mean(BUF)
        median_frame = compute_median(BUF)
        
        # Perform background subtraction
        binary_diff_mean = background_subtraction(gray_frame, mean_frame)
        binary_diff_median = background_subtraction(gray_frame, median_frame)

        # Perform further processing as needed (e.g., median filtering, morphological operations)

        # Display or save results
        cv2.imshow("Mean Background Subtraction", binary_diff_mean)
        cv2.imshow("Median Background Subtraction", binary_diff_median)
        cv2.waitKey(0)  # Wait for a key press before showing next frame

# Example usage
# frames = load_frames()  # Load frames from a video file or camera
# N = 10  # Buffer size
# process_frames(frames, N)
