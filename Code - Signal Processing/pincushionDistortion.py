from functions import *
# Load the image
img = cv2.imread('Pictures/chess.png')

# Display the original and distorted images
cv2.imshow('Original', img)
cv2.imshow('Distorted', distortImage(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
