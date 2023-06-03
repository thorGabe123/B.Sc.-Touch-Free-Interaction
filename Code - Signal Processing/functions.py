import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from scipy.fftpack import ifft2
import pandas as pd
import time
# import websocket
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
# import mouse
from rotpy.system import SpinSystem
from rotpy.camera import CameraList
from rotpy.camera_nodes import CameraNodes
from numpy import unravel_index


def startDriver():
    """Starts the WebDriver (Chrome) and returns the driver object."""
    driver = webdriver.Chrome(executable_path=r"C:\Users\Webdriver\chromedriver.exe")
    return driver

def getToScreener(driver):
    """Navigates the WebDriver to the specified URL and returns the driver object."""
    URL = 'file:///C:/Users/Bruger/Documents/GitHub/B.Sc.-Touch-Free-Interaction/Website/index.html'
#     URL = 'file:///C:/Users/mkrhi/OneDrive/Dokumenter/GitHub/Touch-Free-Interaction/Website/index.html'
    driver.get(URL)
    return driver

def getToDarkness(driver):
    """Navigates the WebDriver to the specified URL and returns the driver object."""
#     URL = 'file:///C:/Users/mkrhi/OneDrive/Dokumenter/GitHub/Touch-Free-Interaction/Website/black.html'
    URL = 'file:///C:/Users/Bruger/Documents/GitHub/B.Sc.-Touch-Free-Interaction/Website/black.html'
    driver.get(URL)
    return driver

def clickNumber(number, driver):
    """Clicks the specified number on a web page using the WebDriver."""
    button = driver.find_element(By.ID, f'key{number}')
    button.click()

def on_message(ws, message):
    """Prints the received message through WebSocket."""
    print(message)

def power_spectrum(fft):
    """Returns the power spectrum of the 2D-Fourier Transform."""
    return np.abs(fft)**2

def FFT(image):
    """Returns the fast Fourier transform after it is centered."""
    fft = np.fft.fft2(image, s=None, axes=(-2, -1), norm=None)
    fft = np.fft.fftshift(fft, axes=None)
    return fft

def inverse_FFT(pwr_spectrum):
    """Computes the inverse Fourier transform of the power spectrum."""
    fft_array = np.fft.ifftshift(pwr_spectrum, axes=None)
    img = np.fft.ifft2(fft_array)
    #img = ifft2(fft_array)
    img = np.abs(img)
    print(np.max(img))
    img[img < 1] = 0
    img = (img - np.min(img)) * (255 / (np.max(img) - np.min(img)))
    return img.astype(np.uint8)

def maskDots(fft_img, X, Y, radius_squared):
    """Masks the dots in the FFT image."""
    r1 = (np.arange(crop_size)[:, None] - Y - crop_size/2) ** 2 + (np.arange(crop_size) - X - crop_size/2) ** 2
    r2 = (np.arange(crop_size)[:, None] + Y - crop_size/2) ** 2 + (np.arange(crop_size) + X - crop_size/2) ** 2
    mask = np.logical_and(r1 > radius_squared, r2 > radius_squared)
    fft_img[mask] = 0
    fft_copy = fft_img.copy()
    return fft_copy

def maskLine(fft_img, X, Y, radius_squared):
    """Masks the line in the FFT image."""
    r = []
    for i in range(10):
        r.append((np.arange(crop_size)[:, None] - Y + (2*Y/10*i) - crop_size/2) ** 2 + (np.arange(crop_size) - X + (2*X/10*i) - crop_size/2) ** 2)
    center = (np.arange(crop_size)[:, None] - crop_size/2) ** 2 + (np.arange(crop_size) - crop_size/2) ** 2
    mask = r[0] > radius_squared
    for i in range(9):
        if i not in [3, 4, 5, 6]:
            mask = np.logical_and(mask, r[i+1] > radius_squared)
    fft_img[mask] = 0
    fft_copy = fft_img.copy()
    return fft_copy

def maskSquare(a, radius, X, Y):
    """Masks the square in the FFT image."""
    X = int(X + crop_size/2)
    Y = int(Y + crop_size/2)
    b = np.array(a.copy(), dtype=np.uint8)
    return b[Y - radius:Y + radius, X - radius:X + radius]

def mask(fft_img, height, width, X, Y, radius_squared):
    """Masks the FFT image."""
    return maskDots(fft_img, height, width, X, Y, radius_squared)

def planeInteraction(peak):
    """Checks if the plane interaction meets a condition."""
    value = findPeakGrid(peak)
    # Paper
#     return value > 5 * 10 ** 6, value
    # Pointing-pen
#     return value > 3 * 10 ** 6, value
    # Finger
    return value > 1 * 10 ** 6, value

def keyAreaPressed(img):
    """Returns the pressed key based on the image."""
    keys, points = 3, 10
    height, width = len(img) / keys, len(img[0]) / keys
    total = []
    for h in range(keys-1, -1, -1):
        for w in range(keys):
            value = 0
            for i in range(points):
                for j in range(points):
                    value += img[int(h*height + height/4 + height/2*(i+1)/points)][int(w*width + width/4 + width/2*(i+1)/points)]
            total.append(value)
    multiply_factor = [0.9, 1, 1.6, 1.5, 1.4, 1.4, 1.6, 1.1, 1.2] # Amplifying each key intensity accorting to testing
    for i in range(len(total)):
        total[i] = total[i] * multiply_factor[i]
    ID = f'key{total.index(max(total))+1}'
    return ID

def lowToHigh(currentB, previousB):
    """Checks if a condition is met by comparing two values."""
    passed = False
    if currentB and not previousB:
        passed = True
    return passed

k1 = -0.5  # pincushion distortion
k2 = 0.0
k3 = 0.0
p1 = 0.0
p2 = 0.0
dist_coeffs = np.array([k1, k2, k3, p1, p2], dtype=np.float32)

def distortImage(img):
    """Distorts the image."""
    focal_length = 300
    center_x = img.shape[1] / 2
    center_y = img.shape[0] / 2
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)
    img_distorted = cv2.undistort(img, camera_matrix, dist_coeffs)
    return img_distorted

def findPeakGrid(a):
    """Finds the peak value in a grid."""
    size = len(a)
    b = np.asarray(a)
    return b.max()

def xyPeak(a):
    """Returns the X and Y coordinates of the peak in a grid."""
    peak = findPeakGrid(a)
    return [peak[0]+X-radius, peak[1]+Y-radius]

def cameraInDarkness(cap):
    """Sets camera settings for darkness conditions."""
    cap.set(cv2.CAP_PROP_EXPOSURE, -2)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 104)
    cap.set(cv2.CAP_PROP_SATURATION, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 255)
    cap.set(cv2.CAP_PROP_SHARPNESS, 165)
    cap.set(cv2.CAP_PROP_FOCUS, 18)
    cap.set(cv2.CAP_PROP_GAIN, 255)
    cap.set(cv2.CAP_PROP_TILT, 0)
    cap.set(cv2.CAP_PROP_PAN, 0)
    cap.set(cv2.CAP_PROP_ZOOM, 390)

def cameraInLight(cap):
    """Sets camera settings for light conditions."""
    cap.set(cv2.CAP_PROP_EXPOSURE, -2)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 230)
    cap.set(cv2.CAP_PROP_CONTRAST, 255)
    cap.set(cv2.CAP_PROP_SHARPNESS, 130)
    cap.set(cv2.CAP_PROP_FOCUS, 18)
    cap.set(cv2.CAP_PROP_GAIN, 60)
    cap.set(cv2.CAP_PROP_TILT, -1)
    cap.set(cv2.CAP_PROP_PAN, 0)
    cap.set(cv2.CAP_PROP_ZOOM, 500)
    return cap

def getComputerCamera():
    """Returns the computer's camera object."""
    return cv2.VideoCapture(0)

def getWebCam():
    """Returns the webcam object."""
    return cv2.VideoCapture(1 + cv2.CAP_DSHOW)

def changeCameraSetting(cap):
    """Changes the camera settings."""
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    return cap

def captureImage(camera):
    """Captures an image from the camera."""
    camera.begin_acquisition()
    image_cam = camera.get_next_image(timeout=5)
    image = image_cam.deep_copy_image(image_cam)
    image_data = image_cam.get_image_data()
    image_height = image.get_height()
    image_width = image.get_width()
    image_cam.release()
    camera.end_acquisition()
    numpy_array = np.array(image_data)
    reshaped_array = numpy_array.reshape((image_height, image_width))
    reshaped_array = reshaped_array
    return resizeImg(reshaped_array)

def initCamera():
    """Initializes the camera."""
    system = SpinSystem()
    cameras = CameraList.create_from_system(system, update_cams=True, update_interfaces=True)
    camera = cameras.create_camera_by_index(0)
    camera.init_cam()
    return camera

def resizeImg(img):
    """Resizes the image."""
    smallest_axis = min(len(img), len(img[0]))
    img = img[100:smallest_axis-100, 100:smallest_axis-100]
    dim = (crop_size, crop_size)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

