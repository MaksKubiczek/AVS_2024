from ImageProcessing.ImageProcessing import ImageProcessing
from utilities.FileManager import FileManager
from utilities.Histrogram import Histrogram
import matplotlib.pyplot as plt



def main() ->None:



    # shapes drawing
    # x = [100, 150, 200, 250]
    # y = [50, 100, 150, 200]
    # plt.plot(x,y,"r.", markersize = 10)


    # *conversion colour* #
    # Image1 = read_from_jpg("mandrill")
    # Image2 = read_from_jpg("mandrill")
    # IG = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("v1", IG)
    # cv2.waitKey(0)
    # IHSV = cv2.cvtColor(Image2, cv2.COLOR_BGR2HSV)
    # cv2.imshow("v2", IHSV)
    # cv2.waitKey(0)

# Wywołanie funkcji read_from_path na instancji FileManager
    lena = FileManager.read_from_path_grey(r"pictures\lena.png")
    mandrill = FileManager.read_from_path_grey(r"pictures\mandrill.jpg")
    add_result = ImageProcessing.images_addition(r"pictures\mandrill.jpg", r"pictures\lena.png")
    sub_result = ImageProcessing.images_substraction(r"pictures\mandrill.jpg", r"pictures\lena.png")
    mul_result = ImageProcessing.images_multiplication(r"pictures\mandrill.jpg", r"pictures\lena.png", 1)
    linear_combination_result = ImageProcessing.images_linear_combination(r"pictures\mandrill.jpg", 0.5, r"pictures\lena.png", 0.5, 0)
    dif_result = ImageProcessing.images_abs_diff(r"pictures\mandrill.jpg", r"pictures\lena.png")


    # # Display results
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 3, 1), plt.imshow(lena, cmap='gray'), plt.title('Lena')
    plt.subplot(3, 3, 2), plt.imshow(mandrill, cmap='gray'), plt.title('Mandrill')
    plt.subplot(3, 3, 3), plt.imshow(add_result, cmap='gray'), plt.title('Addition')
    plt.subplot(3, 3, 4), plt.imshow(sub_result, cmap='gray'), plt.title('Subtraction')
    plt.subplot(3, 3, 5), plt.imshow(mul_result, cmap='gray'), plt.title('Multiplication')
    plt.subplot(3, 3, 6), plt.imshow(linear_combination_result, cmap='gray'), plt.title('Linear Combination')
    plt.subplot(3, 3, 8), plt.imshow(dif_result, cmap='gray'), plt.title('Abs Diff ')

    plt.show()

    lena2 = FileManager.read_from_path_grey(r"pictures\lena.png")
    custom_hist = Histrogram.hist(lena2)
    opencv_hist = Histrogram.cv2_hist(lena2)

    # Plot the histograms for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(custom_hist)
    plt.title('Custom Histogram')

    plt.subplot(1, 2, 2)
    plt.plot(opencv_hist)
    plt.title('OpenCV Histogram')

    plt.show()

if __name__ == "__main__":
    main()
