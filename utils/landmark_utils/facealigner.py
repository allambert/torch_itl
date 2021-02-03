# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import os
import pandas as pd

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
		output_lnd = cv2.transform(shape[np.newaxis], M)
		if image is not None:
			output = cv2.warpAffine(image, M, (w, h),
				flags=cv2.INTER_CUBIC)

			# return the aligned face and lnd
			return output, np.squeeze(output_lnd)
		else:
			return np.squeeze(output_lnd)

#%%

if __name__ == "__main__":

	dataset = 'rafd'
	if dataset == 'try':
		fa = FaceAligner()
		im_path = '/home/mlpboon/Downloads/KDEF_and_AKDEF/KDEF/AF01/AF01NES.JPG'
		lnd_path = '/home/mlpboon/post-doc/repositories/torch_itl/datasets/KDEF/KDEF_LANDMARKS/AF01NES.txt'
		im = cv2.imread(im_path)
		points = np.loadtxt(lnd_path).reshape(68, 2)
		output, lnd = fa.align(im, points)
		for l in range(lnd.shape[0]):
			cv2.circle(output, (int(lnd[l,0]), int(lnd[l,1])), 2, (0, 255, 0), -1)
		cv2.imwrite('aligned_face.jpg', output)
	elif dataset == 'kdef':
		# init aligner

		fa = FaceAligner(desiredFaceWidth=128)

		# get source info
		# Decompose the path into prefix and suffix so that it works independently of the user
		prefix_torch_itl = '/Users/alambert/Recherche/ITL/code/torch_itl'
		prefix_kdef_data = '/Users/alambert/Recherche/ITL/code/KDEF_AND_AKDEF/'
		def shorten_path(path):
			return path[39:]
		source_data_csv = prefix_kdef_data + 'KDEF.csv'
		df = pd.read_csv(source_data_csv)
		df['file_path'] = df['file_path'].apply(shorten_path)
		df_filter = df.loc[df['profile'] == 'S']

		lnd_dir = prefix_torch_itl + '/datasets/KDEF/KDEF_LANDMARKS'

		# destination directory structure
		dest_base_path = '../../datasets'
		dest_parent_folder_path = os.path.join(dest_base_path, 'KDEF_Aligned')
		dest_sub_folder_datapath = os.path.join(dest_parent_folder_path, 'KDEF')
		dest_sub_folder_lndpath = os.path.join(dest_parent_folder_path, 'KDEF_LANDMARKS')

		if not os.path.exists(dest_parent_folder_path):
			os.makedirs(dest_sub_folder_datapath)
			os.makedirs(dest_sub_folder_lndpath)

		for index, row in df_filter.iterrows():

			# read each image and landmarks
			# ----file paths
			im_path = prefix_kdef_data + row['file_path']
			file_name = (prefix_kdef_data + row['file_path']).split('/')[-1].split('.')[0]
			lnd_file_path = os.path.join(lnd_dir, file_name + '.txt')
			# ----read_image
			image = cv2.imread(im_path)
			# get landmarks
			points = np.loadtxt(lnd_file_path).reshape(68, 2)
			# compute alignment
			image_aligned, points_aligned = fa.align(image, points)

			# store back in a folder structure similar to KDEF
			row_datafolder = os.path.join(dest_sub_folder_datapath, row['id'])
			row_impath = os.path.join(row_datafolder, file_name + '.JPG')
			row_lndpath = os.path.join(dest_sub_folder_lndpath, file_name + '.txt')
			if not os.path.exists(row_datafolder):
				os.mkdir(row_datafolder)
			cv2.imwrite(row_impath, image_aligned)
			with open(row_lndpath, 'w') as file:
				for idx_l in range(68):
					file.write("{} {}\n".format(points_aligned[idx_l, 0], points_aligned[idx_l, 1]))
	elif dataset == 'rafd':
		# init face aligner
		fa = FaceAligner(desiredFaceWidth=128)

		# set path prefix
		prefix_torch_itl = '/home/mlpboon/post-doc/repositories/torch_itl'
		prefix_rafd_data = '/home/mlpboon/Downloads/'
		lnd_dir = os.path.join(prefix_torch_itl , 'datasets/Rafd/Rafd_LANDMARKS')
		# get rafd csv and frontal face list
		df = pd.read_csv(os.path.join(prefix_rafd_data, 'Rafd/Rafd.csv'))
		df_filter = df.loc[(df['gaze'] == 'frontal') &
										(df['profile'] == 90)]
		# destination directory structure
		dest_base_path = '../../datasets'
		dest_parent_folder_path = os.path.join(dest_base_path, 'Rafd_Aligned')
		dest_sub_folder_datapath = os.path.join(dest_parent_folder_path, 'Rafd')
		dest_sub_folder_lndpath = os.path.join(dest_parent_folder_path, 'Rafd_LANDMARKS')

		if not os.path.exists(dest_parent_folder_path):
			os.makedirs(dest_sub_folder_datapath)
			os.makedirs(dest_sub_folder_lndpath)

		for index, row in df_filter.iterrows():

			# read each image and landmarks
			# ----file paths
			im_path = prefix_rafd_data + row['file_path']
			file_name = row['file_path'].split('/')[-1].split('.')[0]
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
