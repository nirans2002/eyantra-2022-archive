import cv2
import numpy as np
from collections import defaultdict


def segment_by_angle_kmeans(lines, k=2, **kwargs):

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

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

img = cv2.imread('/home/navneeth/EgoPro/eyantra/gitrepo/eyantra-2022/PB_Task1_Ubuntu/PB_Task1_Ubuntu/Task1A/public_test_images/maze_1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 50, 150, apertureSize=3)


lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
seg = segment_by_angle_kmeans(lines)

intersections = segmented_intersections(seg)


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
def x_coordinate(ele):
    return ele[0]

intersections = sorted(intersections,key=x_coordinate)


def sort_coordinates_with_letter(intersections):
    a=intersections[:28]
    b=intersections[28:56]
    c=intersections[56:84]
    d=intersections[84:112]
    e=intersections[112:140]
    f=intersections[140:168]
    g=intersections[168:196]


    A=[[] for i in range(7)]
    B=[[] for i in range(7)]
    C=[[] for i in range(7)]
    D=[[] for i in range(7)]
    E=[[] for i in range(7)]
    F=[[] for i in range(7)]
    G=[[] for i in range(7)]

    for i in range(0,len(a)//2-1,2):
        index = i//2
        point = a[i][0]
        A[index].append(point)

        n_i = i+14
        point = a[n_i][0]        
        A[index].append(point)

        n_i = i+15
        point = a[n_i][0] 
        A[index].append(point)


        n_i = i+1
        point = a[n_i][0] 
        A[index].append(point)



    for i in range(0,len(b)//2-1,2):
        index = i//2
        point = a[i][0]
        B[index].append(point)


        n_i = i+14
        point = a[n_i][0]        
        B[index].append(point)

        
        n_i = i+15
        point = a[n_i][0] 
        B[index].append(point)



        n_i = i+1
        point = a[n_i][0] 
        B[index].append(point)



    for i in range(0,len(c)//2-1,2):
        index = i//2
        point = a[i][0]
        C[index].append(point)


        n_i = i+14
        point = a[n_i][0]        
        C[index].append(point)

        
        n_i = i+15
        point = a[n_i][0] 
        C[index].append(point)


        n_i = i+1
        point = a[n_i][0] 
        C[index].append(point)



    for i in range(0,len(d)//2-1,2):
        index = i//2
        point = a[i][0]
        D[index].append(point)


        n_i = i+14
        point = a[n_i][0]        
        D[index].append(point)

        
        n_i = i+15
        point = a[n_i][0] 
        D[index].append(point)


        n_i = i+1
        point = a[n_i][0] 
        D[index].append(point)



    for i in range(0,len(e)//2-1,2):
        index = i//2
        point = a[i][0]
        E[index].append(point)


        n_i = i+14
        point = a[n_i][0]        
        E[index].append(point)

        
        n_i = i+15
        point = a[n_i][0] 
        E[index].append(point)


        n_i = i+1
        point = a[n_i][0] 
        E[index].append(point)



    for i in range(0,len(f)//2-1,2):
        index = i//2
        point = a[i][0]
        F[index].append(point)


        n_i = i+14
        point = a[n_i][0]        
        F[index].append(point)

        
        n_i = i+15
        point = a[n_i][0] 
        F[index].append(point)



        n_i = i+1
        point = a[n_i][0] 
        F[index].append(point)

 

    for i in range(0,len(g)//2-1,2):
        index = i//2
        point = a[i][0]
        G[index].append(point)


        n_i = i+14
        point = a[n_i][0]        
        G[index].append(point)

        
        n_i = i+15
        point = a[n_i][0] 
        G[index].append(point)

 

        n_i = i+1
        point = a[n_i][0] 
        G[index].append(point)




    return A,B,C,D,E,F,G

def crop_image(img,y1,y2,x1,x2):
    img=np.array(img[y1:y2,x1:x2,:])
    return img


def detect_signals(img,stream):
    result=[]
    color={'green':1,'blue':0,'red':0}

    
    for i in range(len(stream)):
        max_count=0
        x1,y1 = stream[i][0]
        x2,y2 = stream[i][2]
        cv2.imwrite("crop.jpg",img[x1:x2,y1:y2,:])       
        lower_green = np.array([0, 255, 0], dtype = "uint8") 
        upper_green= np.array([0, 255, 0], dtype = "uint8")
        mask = cv2.inRange(img[x1:x2,y1:y2:], lower_green, upper_green)
        # res = cv2.bitwise_and(img,img, mask= mask)
 


    return result
A,B,C,D,E,F,G = sort_coordinates_with_letter(intersections)
co = detect_signals(img,A)
print(co)

for i in range(len(A)):
    # print(A[i])
    x1,y1 = A[i][0]
    x2,y2 = A[i][2]
    print(x1,x2,y1,y2)
    # cv2.imwrite("crop.jpg",img[x1:x2,y1:y2,:])
    # x,y=a
    # img = cv2.circle(img, (x,y), 3, (255,0,0), 3)
    # i+=1

cv2.imwrite('linesDetected.jpg', img)