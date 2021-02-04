# import the necessary packages
from collections import OrderedDict
import subprocess
import numpy as np
import cv2
import os


# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


class FaceAligner:
	"""
	Modified from python library imutils
	"""
	def __init__(self, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height

		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth

	def align(self, image, shape):
		# tight_aux = 10
		# w = image.shape[1]
		# h = image.shape[0]
		# min_x = int(np.maximum(np.round(np.min(shape[:, 0])) - tight_aux, 0))
		# min_y = int(np.maximum(np.round(np.min(shape[:, 1])) - tight_aux, 0))
		# max_x = int(np.minimum(np.round(np.max(shape[:, 0])) + tight_aux, w - 1))
		# max_y = int(np.minimum(np.round(np.max(shape[:, 1])) + tight_aux, h - 1))
		# shape[:, 0] = shape[:, 0] - min_x
		# shape[:, 1] = shape[:, 1] - min_y

		if len(shape) == 68:
			# extract the left and right eye (x, y)-coordinates
			(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)
		output_lnd = cv2.transform(shape[np.newaxis], M)
		# return the aligned face and lnd
		return output, np.squeeze(output_lnd)


def list_and_landmarks(dataset, data_dir, lnd_dir, paths_txt, predictor_path):

	# generate frontal faces list according to dataset
	img_list = []
	if dataset == 'KDEF':
		for dir, subdir, filenames in os.walk(data_dir):
			for f in filenames:
				if f.endswith('S.JPG'):
					img_list.append(os.path.join(dir, f))
	elif dataset == 'Rafd':
		for img in sorted(os.listdir(data_dir)):
			print(img)
			if img.split('.')[-1].lower() == 'jpg':
				fname_part = img.split('.')[0].split('_')
				print(fname_part)
				if fname_part[5] == 'frontal' and fname_part[0][4:] == '090':
					img_list.append(os.path.join(data_dir, img))

	with open(paths_txt, 'w') as f:
		for fpath in img_list:
			f.write('{}\n'.format(fpath))

	# get landmarks from dlib
	if not os.path.exists(predictor_path):
		print('Landmarks predictor does not exist.')
	else:
		dlib_cmd = ' '.join(['python dlib_landmarks.py', predictor_path, paths_txt, lnd_dir])
		with subprocess.Popen(dlib_cmd, shell=True, stdout=subprocess.PIPE) as cmd:
			for line in cmd.stdout:
				print(line)
	return img_list


def data_preprocess(dataset, data_dir, dest_base_dir, predictor_path):

	"""
	Parameters
	----------
	dataset: str
			KDEF or Rafd, dataset names
	data_dir: str
			path to data directory to fetch images
	dest_base_dir:
			directory to store all outputs
	predictor_path:
			predictor for computing dlib landmarks
			Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

	Returns
	-------
	Saves all dlib landmarks, aligned images/landmarks in dedicated folders on dest_base_dir
	"""

	print('COMPUTING LANDMARKS')
	# create file list, compute landmarks and return with list for alignment
	lnd_dir = os.path.join(dest_base_dir, dataset + '_LANDMARKS')
	paths_txt = os.path.join(dest_base_dir, dataset + '_frontal_list.txt')
	if not os.path.exists(lnd_dir):
		os.makedirs(lnd_dir)
	img_list = list_and_landmarks(dataset, data_dir, lnd_dir, paths_txt, predictor_path)

	print('PERFORMING ALIGNMENT')
	# init aligner
	fa = FaceAligner(desiredFaceWidth=128)

	# destination directory structure
	dest_parent_folder_path = os.path.join(dest_base_dir, dataset + '_Aligned')
	dest_sub_folder_datapath = os.path.join(dest_parent_folder_path, dataset)
	dest_sub_folder_lndpath = os.path.join(dest_parent_folder_path, dataset + '_LANDMARKS')

	if not os.path.exists(dest_parent_folder_path):
		os.makedirs(dest_sub_folder_datapath)
		os.makedirs(dest_sub_folder_lndpath)

	for index, row in enumerate(img_list):

		# read each image and landmarks
		# ----file paths
		im_path = row
		file_name = row.split('/')[-1].split('.')[0]
		lnd_file_path = os.path.join(lnd_dir, file_name + '.txt')

		# ----read_image
		image = cv2.imread(im_path)

		# get landmarks
		points = np.loadtxt(lnd_file_path).reshape(68, 2)

		# compute alignment
		image_aligned, points_aligned = fa.align(image, points)

		# store back in a folder structure similar to Rafd
		row_impath = os.path.join(dest_sub_folder_datapath, file_name + '.JPG')
		row_lndpath = os.path.join(dest_sub_folder_lndpath, file_name + '.txt')

		cv2.imwrite(row_impath, image_aligned)
		with open(row_lndpath, 'w') as file:
			for idx_l in range(68):
				file.write("{} {}\n".format(points_aligned[idx_l, 0], points_aligned[idx_l, 1]))


if __name__ == "__main__":
	# download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	predictor_path = 'shape_predictor_68_face_landmarks.dat'

	dataset = 'KDEF'
	data_dir = 'PATH_TO_KDEF'
	dest_dir = 'DESTINATION PATH'

	# dataset = 'Rafd'
	# data_dir = 'PATH_TO_Rafd'
	# dest_dir = 'DESTINATION PATH'

	# change dest dir to be compatible with other scripts and store in right location
	# dest_dir = '../../../torch_itl/datasets'

	data_preprocess(dataset, data_dir, dest_dir, predictor_path)