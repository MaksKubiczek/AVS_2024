import cv2
import numpy as np

# Terminology
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
width = 9
height = 6
square_size = 0.025
objp = np.zeros((height * width, 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
objp = objp * square_size
objpoints = []
imgpointsLeft = []
imgpointsRight = []
img_width = 640
img_height = 480
image_size = (img_width, img_height)
K_left = np . zeros ((3 , 3) )
D_left = np . zeros ((4 , 1) )
K_right = np . zeros ((3 , 3) )
D_right = np . zeros ((4 , 1) )

path = "resources/pairs/"
number_of_images = 10  # Assuming you have 10 images in the sequence

# Collect object points and image points for each pair of images
for i in range(1, number_of_images + 1):
    img_left = cv2.imread(path + "left_%02d.png" % i)
    img_right = cv2.imread(path + "right_%02d.png" % i)
    
    # Convert images to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (width, height), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, height), None)
    
    if ret_left and ret_right:
        objpoints.append(objp)
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (3, 3), (-1, -1), criteria)
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (3, 3), (-1, -1), criteria)
        imgpointsLeft.append(corners2_left)
        imgpointsRight.append(corners2_right)
    else:
        print("Chessboard couldn't be detected. Image pair:", i)

# Convert lists to arrays
objpoints = np.asarray(objpoints, dtype=np.float64)
imgpointsLeft = np.asarray(imgpointsLeft, dtype=np.float64)
imgpointsRight = np.asarray(imgpointsRight, dtype=np.float64)

# Perform stereo calibration
RMS, _, _, _, _, rotationMatrix, translationVector = cv2.fisheye.stereoCalibrate(
    objpoints, imgpointsLeft, imgpointsRight,
    K_left, D_left,
    K_right, D_right,
    image_size, None, None,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
)

# Initialize matrices for rectification
R2 = np.zeros([3, 3])
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
Q = np.zeros([4, 4])

# Stereo rectification
(leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap) = cv2.fisheye.stereoRectify(
    K_left, D_left,
    K_right, D_right,
    image_size,
    rotationMatrix, translationVector,
    0, R2, P1, P2, Q,
    cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0
)

# Initialize undistortion and rectification maps for left and right images
map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, leftRectification,
    leftProjection, image_size, cv2.CV_16SC2
)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, rightRectification,
    rightProjection, image_size, cv2.CV_16SC2
)

# Remap images to remove distortion
for i in range(1, number_of_images + 1):
    img_left = cv2.imread(path + "left_%02d.png" % i)
    img_right = cv2.imread(path + "right_%02d.png" % i)
    
    dst_L = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    dst_R = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)
    
    # Display the rectified images with horizontal lines
    N, XX, YY = dst_L.shape[::-1]
    visRectify = np.zeros((YY, XX * 2, N), np.uint8)
    visRectify[:, 0:XX, :] = dst_L
    visRectify[:, XX:XX * 2, :] = dst_R
    for y in range(0, YY, 10):
        cv2.line(visRectify, (0, y), (XX * 2, y), (255, 0, 0))
    
    cv2.imshow('Rectified Images with Horizontal Lines', visRectify)
    cv2.waitKey(0)

cv2.destroyAllWindows()
