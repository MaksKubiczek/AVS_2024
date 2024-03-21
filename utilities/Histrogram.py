import numpy as np
import cv2

class Histrogram:
    def hist (image):
        h = np.zeros ((256 ,1) , np.float32 ) 
        height , width = image.shape [:2] 
        for y in range (height):
            for x in range (width):
                pixel_value = image[x, y]
                h[pixel_value] += 1
        
        return h

    def cv2_hist(image):
        return cv2.calcHist([image], [0], None, [256], [0, 256])