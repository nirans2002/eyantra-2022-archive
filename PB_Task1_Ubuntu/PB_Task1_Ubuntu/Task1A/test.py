import cv2
import numpy as np
from collections import defaultdict


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)

    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def colour(i):
	if i==0:
		return (0,0,255)
	elif i==1:
		return (0,255,0)

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

img = cv2.imread('/home/navneeth/EgoPro/eyantra/gitrepo/eyantra-2022/PB_Task1_Ubuntu/PB_Task1_Ubuntu/Task1A/public_test_images/maze_2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 50, 150, apertureSize=3)


lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
seg = segment_by_angle_kmeans(lines)

intersections = segmented_intersections(seg)
print(len(intersections))

for i in range(len(seg)):
	for r_theta in seg[i]:
		arr = np.array(r_theta[0], dtype=np.float64)
		r, theta = arr
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*r
		y0 = b*r
	
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(img, (x1, y1), (x2, y2), colour(i), 2)
i=0
intersections.sort
for point in intersections[80:80+29]:
	x,y=point[0]
	img = cv2.circle(img, (x,y), 3, (255,0,0), 3)
	i+=1

cv2.imwrite('linesDetected.jpg', img)