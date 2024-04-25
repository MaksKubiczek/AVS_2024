import numpy as np
import cv2
from ImageProcessing.Background import Backgroud
from utilities.FileManager import FileManager

class Lab3:
    buffer_size = 60
    alpha = 0.01

    def lab3():
        # Init buffer and counter
        BUF=[]
        iN = 0

        # Main loop
        for i in range(300, 1100, 2):
            file_path = f"resources/highway/input/in{i:06d}.jpg"
            current_frame = cv2.imread(file_path)

            # Convert to grayscale
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Store current frame in buffer
            if len(BUF) < Lab3.buffer_size:
                BUF.append(current_frame_gray)
            else:
                BUF[iN] = current_frame_gray

            # Increment counter and reset if necessary
            iN += 1
            if iN == Lab3.buffer_size:
                iN = 0

            # Calculate mean or median from buffer
            buffer_mean = np.mean(BUF, axis=0).astype(np.uint8)
            buffer_median = np.median(BUF, axis=0).astype(np.uint8)

            # Perform background subtraction using mean and median models
            result_mean = Backgroud.background_subtraction(current_frame, buffer_mean)
            result_median = Backgroud.background_subtraction(current_frame, buffer_median)

            # Display results or save them to files, perform further analysis as needed
            cv2.imshow('Result Mean', result_mean)
            cv2.imshow('Result Median', result_median)
            cv2.waitKey(30)
        
            # Update background model using mean approximation
            buffer_mean = Backgroud.update_background_mean(current_frame_gray, buffer_mean, Lab3.alpha)
    
            # Update background model using median approximation
            buffer_median = Backgroud.update_background_median(current_frame_gray, buffer_median)

            # Perform conservative update of background model
            if i > 0:
                previous_mask = result_mean > 0  # Assume result_mean contains previous object mask
            
                # Ensure buffer_mean is defined before using it
                if 'buffer_mean' in locals():
                    # Convert buffer_mean to the same type as current_frame_gray
                    buffer_mean_same_type = buffer_mean.astype(current_frame_gray.dtype)
                    buffer_mean = Backgroud.update_background_conservative(current_frame_gray, buffer_mean_same_type, Lab3.alpha, previous_mask)
                else:
                    # Handle case when buffer_mean is not defined (e.g., first iteration)
                    buffer_mean = Backgroud.update_background_conservative(current_frame_gray, buffer_mean, Lab3.alpha, previous_mask)
    
        cv2.destroyAllWindows()