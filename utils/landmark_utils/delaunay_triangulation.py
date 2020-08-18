#!/usr/bin/python

import cv2
import sys

image_path = sys.argv[1]
output_folder = sys.argv[2]


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color ):
    cv2.circle(img, p, 2, color, cv2.cv.CV_FILLED, cv2.LINE_AA, 0)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, dictionary1):

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    list1 = []
    for t in triangleList:
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            list1.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))

    return list1


if __name__ == '__main__':

    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    # Turn on animation while drawing triangles
    animate = False
    
    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    # Read in the image.
    im_path = image_path
    im_name = im_path.split('/')[-1].split('.')[0]
    img = cv2.imread(im_path)
    
    # Keep a copy around
    img_orig = img.copy()
    
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Create an array of points.
    points = []
    
    # Read in the points from a text file
    with open(im_name+".txt") as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
        
        # Show animation
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 255, 255))
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    dictionary = {p[0]: p[1] for p in list(zip(points, range(68)))}
    # Draw delaunay triangles
    del_triangles = draw_delaunay(img, subdiv, dictionary);
    with open("del_triangles_" + im_name + '.txt', 'w') as file:
        for v in range(len(del_triangles)):
            file.write("{} {} {}\n".format(del_triangles[v][0],del_triangles[v][1], del_triangles[v][2]))
