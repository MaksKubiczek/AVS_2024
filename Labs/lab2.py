from ImageProcessing.ImageOperation import ImageProcessing
from utilities.FileManager import FileManager
import matplotlib.pyplot as plt
import cv2
import numpy as np

# lab2

''' Stages of Algorithm:

    > input image

    > boundary boxes

    > binarization

    > median filtering

    > morphological operations

    > representing labels

    > comparing with reference image - ground truth

'''

class Lab2:
    
    step_value = 1 # have to be int
    threshold = 50
    kernel = np.ones((3, 3), np.uint8)
    erosion_iteration = 3
    dilation_iteration = 3
    minimal_rectangle_dimension = 10

    @staticmethod
    def set_parameters(new_step_value, new_threshold, new_kernel, new_erosion_iteration, new_dilation_iteration, new_minimal_rectangle_dimension):
        Lab2.step_value = new_step_value
        Lab2.threshold = new_threshold
        Lab2.kernel = new_kernel
        Lab2.erosion_iteration = new_erosion_iteration
        Lab2.dilation_iteration = new_dilation_iteration
        Lab2.minimal_rectangle_dimension = new_minimal_rectangle_dimension

    @staticmethod
    def set_step_value(new_step_value):
        Lab2.step_value = new_step_value

    @staticmethod
    def set_threshold(new_threshold):
        Lab2.threshold = new_threshold

    @staticmethod
    def set_kernel(new_kernel):
        Lab2.kernel = new_kernel

    @staticmethod
    def lab2():
        # Inicjalizacja detektora ruchu MOG
        mog = cv2.createBackgroundSubtractorMOG2()

        for i in range(300, 1100, Lab2.step_value):
            file_path = f"resources/pedestrian/input/in{i:06d}.jpg"
            current_frame = FileManager.read_from_path_grey(file_path)

            # Usunięcie tła
            fg_mask = mog.apply(current_frame)

            # Binaryzacja obrazu
            _, thresh = cv2.threshold(fg_mask, Lab2.threshold, 255, cv2.THRESH_BINARY)

            # Znalezienie konturów
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Dodanie prostokątów wokół konturów z odpowiednimi rozmiarami
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= Lab2.minimal_rectangle_dimension and h >= Lab2.minimal_rectangle_dimension:
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 255), 2)

            #Median filetering
            median_filtered = cv2.medianBlur(thresh, 5)

            #Morphological operations
            erosion = ImageProcessing.erosion(median_filtered, Lab2.kernel, Lab2.erosion_iteration)
            dilation = ImageProcessing.dilation(erosion, Lab2.kernel, Lab2.dilation_iteration)

            #Representating labels
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation, connectivity=8)
            if np.max(labels) > 0:
                label_hue = np.uint8(200 * labels / np.max(labels))
                blank_ch = 255 * np.ones_like(label_hue)
                labeled_image = cv2.merge([label_hue, blank_ch, blank_ch])
                labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_HSV2BGR)
                labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2GRAY)
                labeled_image[label_hue == 0] = 0

                # Wyświetlenie obrazu z prostokątami
                cv2.imshow("boundary box", current_frame)
                cv2.imshow("binarize", thresh)
                cv2.imshow("median filtering", median_filtered)
                cv2.imshow("erosion", erosion)
                cv2.imshow("dilation", dilation)
                cv2.imshow("labels", labeled_image)

            cv2.waitKey(10)