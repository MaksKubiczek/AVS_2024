import numpy as np
import cv2

class Histrogram:
    def hist (Image):
        h = np.zeros ((256 ,1) , np.float32 ) 
        height , width = Image.shape [:2] 
        for y in range (height):
            for x in range (width):
                pixel_value = Image[x, y]
                h[pixel_value] += 1
        
        return h

    def cv2_hist(Image):
        return cv2.calcHist([Image], [0], None, [256], [0, 256])