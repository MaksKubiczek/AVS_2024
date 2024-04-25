import cv2
import numpy as np

class Backgroud:

    def background_subtraction(frame, background_model):
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Compute absolute difference between current frame and background model
        frame_diff = cv2.absdiff(frame_gray, background_model)
    
        # Binarize the difference
        _, binary_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
        # Apply median filtering to smooth the binary image
        binary_diff_filtered = cv2.medianBlur(binary_diff, 5)
    
        # Perform morphological operations to further refine the binary image
        kernel = np.ones((5,5), np.uint8)
        binary_diff_morph = cv2.morphologyEx(binary_diff_filtered, cv2.MORPH_CLOSE, kernel)
    
        return binary_diff_morph

    # Function for mean background model approximation
    def update_background_mean(current_frame_gray, buffer_mean, alpha):
        return alpha * current_frame_gray + (1 - alpha) * buffer_mean

    # Function for median background model approximation
    def update_background_median(current_frame_gray, buffer_median):
        buffer_diff = current_frame_gray - buffer_median
        return np.where(buffer_diff > 0, buffer_median + 1, np.where(buffer_diff < 0, buffer_median - 1, buffer_median))

    # Function for conservative update of background model
    def update_background_conservative(current_frame_gray, background_model, alpha, previous_mask):
        # Compute absolute difference between current frame and background model
        frame_diff = cv2.absdiff(current_frame_gray, background_model)
    
        # Binarize the difference
        _, binary_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
        # Use previous mask for conservative update
        conservative_diff = np.where(previous_mask, 0, binary_diff)
    
        # Update background model using conservative approach
        background_model_updated = alpha * current_frame_gray + (1 - alpha) * conservative_diff
    
        return background_model_updated