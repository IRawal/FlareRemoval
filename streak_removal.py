## Remove streaky lens flare from images
### To run: python streak_removal.py --image [path to image]

import argparse
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
args = vars(ap.parse_args())
# load the image
image = cv2.imread(args["image"])

# convert image to grayscale and blur it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# threshold the image to reveal light regions
thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove noise
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# use Canny edge detector to find the edges in the image
edges = cv2.Canny(blurred, 70, 140)

# Use Hough Probabilistic Line Transform to identify straight edges
rho = 1
theta = np.pi/180
lines = cv2.HoughLinesP(edges, rho, theta, threshold=1, minLineLength=50, maxLineGap=5); 

# divide image into a 16x16 grid
height, width = image.shape[:2]
subframe_height = height // 16
subframe_width = width // 16

# find the coordinates of each sub-frame
frame_coordinates = []
for i in range(16):
    for j in range(16):
        x_min = j * subframe_width
        y_min = i * subframe_height
        x_max = (j + 1) * subframe_width
        y_max = (i + 1) * subframe_height
        frame_coordinates.append((x_min, y_min, x_max, y_max))

# cluster by Frame
frame_clusters = []
for line in lines:
    for idx, frame_coord in enumerate(frame_coordinates):
        x1, y1, x2, y2 = line[0]
        if (frame_coord[0] <= x1 <= frame_coord[2]) and (frame_coord[1] <= y1 <= frame_coord[3]):
            frame_clusters.append(idx)
            break

# find slopes of the frame clustered lines
slope_clusters = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
    slope_clusters.append(slope)

# cluster by collinearity
collinearity_clusters = []
for i, line1 in enumerate(lines):
    for j, line2 in enumerate(lines):
        if i != j:
            x1, y1, x2, y2 = line1[0]
            xx1, yy1, xx2, yy2 = line2[0]
            area = abs((x2 - x1) * (yy2 - y1) - (y2 - y1) * (xx2 - x1))
            if area < 2:
                collinearity_clusters.append((i, j))

# select the outermost extreme coordinates
streaks = []
for cluster in set(frame_clusters):
    lines_in_cluster = [lines[i][0] for i, c in enumerate(frame_clusters) if c == cluster]
    outermost_coords = [
        min(lines_in_cluster, key=lambda x: min(x[0], x[2])),
        max(lines_in_cluster, key=lambda x: max(x[0], x[2])),
        min(lines_in_cluster, key=lambda x: min(x[1], x[3])),
        max(lines_in_cluster, key=lambda x: max(x[1], x[3]))
    ]
    streaks.append(outermost_coords)

# draw the streaks
if streaks is not None:
    for streak in streaks:
        x1, y1, x2, y2 = streak[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# show streaks
cv2.imshow("streaks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# create a blank mask image
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# draw lines on the mask
for line in streaks:
    x1, y1, x2, y2 = line[0]
    cv2.line(mask, (x1, y1), (x2, y2), (255), thickness=10)  # Draw lines on the mask

# fill the areas between the lines to create the mask
lines_as_points = [np.array([[x1, y1], [x2, y2]], dtype=np.int32) for x1, y1, x2, y2 in lines[0]]
mask = cv2.fillPoly(mask, pts=lines_as_points, color=(255))

# show the mask
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# inpaint image
inpainted = cv2.inpaint( image, mask, 3, cv2.INPAINT_TELEA)

# show results
cv2.imshow("Inpainted", inpainted)
cv2.waitKey(0)
