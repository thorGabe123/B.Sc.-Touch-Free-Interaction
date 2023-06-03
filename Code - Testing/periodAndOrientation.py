from functions import *

def main():
    camera = initCamera()

    X1, Y1, radius, radius_squared = 20,0,20, 300
    crop_size = 300

    # User values
    wavelength = 100
    angle = 1/4*np.pi


    # Generate meshgrid
    x = np.arange(-405, 405, 1)
    X, Y = np.meshgrid(x, x)
    img = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength
    )

    mm = 3.6 # Valgt ud fra test

    cv2.namedWindow('sliders')
    cv2.imshow('sliders',img)
    cv2.createTrackbar('wavelength', "sliders", 20, 50, on_change)
    cv2.createTrackbar('angle', "sliders", 45, 180, on_change)

    cv2.namedWindow('image')
        
    previousClicked = False
    firstRun = True
    firstImageCaptured = False

    while True:
        cv2.imshow('image',img)
        
        trackSlider = cv2.getTrackbarPos("wavelength","sliders")
        wavelength = trackSlider * mm
        angleSlider = cv2.getTrackbarPos("angle","sliders")
        angle = angleSlider / 100
        img = np.sin(
        2*np.pi*(X*np.cos(angle/180*314) + Y*np.sin(angle/180*314)) / wavelength
        )
        blabla, img = cv2.threshold(img,0.7,1, cv2.THRESH_BINARY)
        
        # Capture frame-by-frame
        frame = captureImage(camera)
        frame = frame[0:crop_size, 0:crop_size]

        # Convert the frame to grayscale
        gray = frame
        
        while firstRun:
            first_gray = gray
            firstRun = False

        # Display the original and transformed frames
        cv2.imshow('Current', gray)
        power_bitch = power_spectrum(FFT(gray))
        cv2.imshow('Magnitude Spectrum', power_bitch)
        
        
        dist_to_center = int(crop_size/2)
        part_of_array = power_bitch[dist_to_center:dist_to_center+40,dist_to_center:dist_to_center+40]
        
        peak = findPeakGrid(part_of_array)
        
        print(part_of_array[peak[1]][peak[0]], end="\r")
        
        if cv2.waitKey(1) & 0xFF == ord('p'):
            printImg(f"{trackSlider}_{angleSlider}",power_bitch)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()