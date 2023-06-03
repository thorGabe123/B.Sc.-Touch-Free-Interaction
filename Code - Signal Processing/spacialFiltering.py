from functions import *

def main():
    # Mask values to be edited
    ####  EDITABLE VALUES  ##################
    Y, X, radius = 7, 13, 6                 #
    radius_squared = int(radius ** 2)       #
    crop_size = 300                         #
    #########################################

    # Open the keyboard using selenium
    driver = startDriver()
    keyboard = getToDarkness(driver)

    previousClicked = False
    firstRun = True
    firstImageCaptured = False

    camera = initCamera()

    while True:
        # Capture the first image 
        while firstRun:
            first_gray = captureImage(camera)
            firstRun = False
            
        gray = captureImage(camera)
        
        # Capture background frame when user presses 'f'
        if cv2.waitKey(1) & 0xFF == ord('f'):
            first_gray = gray
            keyboard = getToScreener(driver)
            firstImageCaptured = True

        # Difference from originally captured image and applied pincushion distortion
        diff_img = cv2.absdiff(gray, first_gray)
        dist_img = distortImage(diff_img)

        # Processing image
        fft_dist = FFT(dist_img)
        masked_img = maskDots(fft_dist, X, Y, radius_squared)
        final_img = inverse_FFT(masked_img)
        ps_mask = power_spectrum(masked_img)
        
        # Show the live signal processing
        cv2.imshow("gray",gray)
        cv2.imshow("power_spectrum",power_spectrum(FFT(gray)))
        cv2.imshow("difference",dist_img)
        cv2.imshow("dist_power",power_spectrum(fft_dist))
        cv2.imshow("final_img",final_img)
        cv2.imshow("masked_power",ps_mask)
        
        
        # Tell if plane is interacted with and which key is pressed
        if firstImageCaptured:
            currentClicked = planeInteraction(ps_mask)[0]
            #print(planeInteraction(ps_mask)[1], end="\r") # Uncomment for testing
            if currentClicked:
                if lowToHigh(currentClicked, previousClicked):
                    keyboard.find_element(By.ID, keyAreaPressed(final_img)).click()
                    #print(keyAreaPressed(final_img), end="\r") # Uncomment for testing
            previousClicked = currentClicked

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    camera.deinit_cam()
    camera.release()
    cv2.destroyAllWindows()
    keyboard.close()

if __name__ == "__main__":
    main()