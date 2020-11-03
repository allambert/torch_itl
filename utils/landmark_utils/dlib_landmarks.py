import sys
import os
import dlib
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

if len(sys.argv) != 5:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_paths_file = sys.argv[2]
output_folder = sys.argv[3]
draw_bool = sys.argv[4]
print(draw_bool)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

if len(faces_paths_file.split('/')[-1].split('.')) == 1:
    with open(faces_paths_file, 'r') as file:
        file_list = [line.rstrip('\n') for line in file]
else:
    file_list = [faces_paths_file]

for f in file_list:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    orig_image = Image.open(f).convert('RGB')
    # win.clear_overlay()
    # win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    assert(len(dets) == 1)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        landmarks = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(landmarks.part(0),
                                                  landmarks.part(1)))
        # Draw the face landmarks on the screen.
        # win.add_overlay(shape)
    with open(os.path.join(output_folder, f.split('/')[-1].split('.')[0] + '.txt'), 'w') as file:
        for pn in range(68):
            file.write("{} {}\n".format(landmarks.part(pn).x,landmarks.part(pn).y)) 
    # win.add_overlay(dets)
    if draw_bool==1:
        draw = ImageDraw.Draw(orig_image)
        colors = [(0, 0, 256)]
        for img_index_i in range(0, 68):
            draw.line([landmarks.part(img_index_i).x, landmarks.part(img_index_i).y-4, landmarks.part(img_index_i).x, landmarks.part(img_index_i).y+4], fill=colors[0 % 5])
            draw.line([landmarks.part(img_index_i).x-4, landmarks.part(img_index_i).y, landmarks.part(img_index_i).x+4, landmarks.part(img_index_i).y], fill=colors[0 % 5])
        plt.imshow(orig_image)
        plt.show()
        dlib.hit_enter_to_continue()

