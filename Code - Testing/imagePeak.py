from functions import *

def main():
    path = "C:/Users/mkrhi/OneDrive/Dokumenter/GitHub/Touch-Free-Interaction/Website/testResultsFingerDetection"'

    crop_size = 300

    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            file_path = os.path.join(path, filename)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[0:crop_size, 0:crop_size]
            
            fft = FFT(image)
            ps = power_spectrum(fft)

            text = filename
            
            radius_squared = 20
            radius_dots = -11
            angle = 150
            
            X = radius_dots * math.sin(math.pi * 2 * angle / 360 + math.pi/2)
            Y = radius_dots * math.cos(math.pi * 2 * angle / 360 + math.pi/2)
            
            ps_mask = ps.copy()
            ps_mask = maskDots(ps_mask, X, Y, radius_squared)
            
            avgval = np.mean(ps_mask, axis=1)
            avgval = np.mean(avgval)
            
            maxval = np.amax(ps_mask)
            
            maxvalpos = findPeakGrid(ps_mask)
            
            print(int(avgval))
            
            cv2.imshow(filename, showImage(ps, X, Y, radius_squared))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()