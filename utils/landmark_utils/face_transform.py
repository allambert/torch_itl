import numpy as np
import cv2
from PIL import Image


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img, t1, t):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = warpImage1

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def makeMorphs(theImage1, theList1, theList2, theList4):

    # Read images
    img1 = theImage1
    
    # Convert Mat to float data type
    img1 = np.float32(img1)

    # Read array of corresponding points
    points1 = theList1
    points = theList2
    
    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    # Read triangles from delaunay_output.txt
    for i in range(len(theList4)):    
        x = int(theList4[i][0])
        y = int(theList4[i][1])
        z = int(theList4[i][2])
            
        t1 = [points1[x], points1[y], points1[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(img1, imgMorph, t1, t)

    temp_res=cv2.cvtColor(np.uint8(imgMorph),cv2.COLOR_BGR2RGB)
    res=Image.fromarray(temp_res)
    return res


if __name__ == "__main__":
    # im_path = "/home/mlpboon/Downloads/KDEF_and_AKDEF/KDEF/AF01/AF01NES.JPG"

    im_path = "jayneel.jpg"
    neu_lndmrks_path = "jayneel.txt"
    trans_lndmrks_path = "pred_relJAYHAS.txt"
    del_triangles_path = "del_triangles_jayneel.txt"

    theImage1 = cv2.imread(im_path)

    # Create an array of points.
    points1 = []
    
    # Read in the points from a text file
    with open(neu_lndmrks_path) as file:
        for line in file :
            x, y = line.split()
            points1.append((int(x), int(y)))

    # Create an array of points.
    points2 = []
    
    # Read in the points from a text file
    with open(trans_lndmrks_path) as file:
        for line in file :
            x, y = line.split()
            points2.append((int(x), int(y)))
    
    # Create an array of points.
    list4 = []
    
    # Read in the points from a text file
    with open(del_triangles_path) as file:
        for line in file:
            v1, v2, v3 = line.split()
            list4.append((int(v1), int(v2), int(v3)))
    res = makeMorphs(theImage1, points1, points2, list4)
