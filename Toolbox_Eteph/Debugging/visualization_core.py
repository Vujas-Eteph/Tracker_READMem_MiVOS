# DOC:
# Collection of function for visualizing results
# Support: Mask
# TODO: Add also BBOx and other types if necessary
# 
# by Eteph-Vujas

import os
import numpy as np
from PIL import Image
import cv2
from matplotlib.pyplot import plot
from scipy import ndimage

import glob


def read_file_with_PIL(path_2_img:str)-> np.array:
	return np.array(Image.open(path_2_img))

def read_file_with_CV():
	pass

def extract_images_from_folder(path_2_folder_containing_images:str) -> np.array:
	list_path_2_images = [os.path.join(path_2_folder_containing_images, elem) for elem in sorted(os.listdir(path_2_folder_containing_images))]
	images_in_array_format = np.array([read_file_with_PIL(path_2_an_image) for path_2_an_image in list_path_2_images])

	return images_in_array_format, list_path_2_images

def create_color_palette(path_2_palette:str) -> np.array:
	palette = Image.open(os.path.expanduser(path_2_palette)).getpalette()
	red_index = [i * 3 + 0 for i in range(0, int(len(palette) / 3))]
	blue_index = [i * 3 + 1 for i in range(0, int(len(palette) / 3))]
	green_index = [i * 3 + 2 for i in range(0, int(len(palette) / 3))]

	red_channel = np.array(palette)[red_index]
	blue_channel = np.array(palette)[blue_index]
	green_channel = np.array(palette)[green_index]

	red_channel = np.expand_dims(red_channel, axis=1)
	blue_channel = np.expand_dims(blue_channel, axis=1)
	green_channel = np.expand_dims(green_channel, axis=1)

	color_palette_for_arrays = np.concatenate((red_channel, blue_channel, green_channel), axis=1)

	return color_palette_for_arrays


def color_mask_with_palette(mask:np.array, palette:np.array) -> np.array:
	mask_with_palette = np.zeros([*mask.shape,3]).astype(np.int)
	for obj_idx in range(mask.min(), mask.max()+1):
		if obj_idx == 0:
			continue
		mask_with_palette[mask==obj_idx] = palette[obj_idx]

	return mask_with_palette

def combine_2_masks_into_1(mask_1:np.array,mask_2:np.array):
	mask_output = mask_1.copy()
	M2 = mask_2.sum(axis=-1) != 0
	DIFF = (mask_1.sum(axis=-1) + mask_2.sum(axis=-1)) > 255

	mask_output[M2] = mask_2[M2]
	mask_output[DIFF] = np.array([2,215,214])
	# mask_output[DIFF] = np.array([255,255,0]) # for the GT

	return mask_output

def combine_img_with_mask(img:np.array,mask:np.array, alpha=0.6):
	img_2_draw = img.copy()
	H,W,_ = np.where(mask != 0)
	for channel in range(0,img.shape[-1]):
		img_2_draw[:,:,channel] = (1-alpha)*img[:,:,channel] + alpha*mask[:,:,channel]
		# img_2_draw[:,:,channel][mask[:,:,1] != 0] = (1 - aplha) * img[:, :, channel][mask[:,:,1] != 0] + aplha * mask[:, :, channel][mask[:,:,1] != 0]


	return img_2_draw


def extract_contour(mask:np.array) -> np.array:
	mask_contour = np.zeros([*mask.shape]).astype(mask.dtype)
	for obj_idx in np.unique(mask)[1:]:
		eroded_mask = ndimage.binary_erosion(mask==obj_idx, iterations=4)
		# eroded_mask = ndimage.binary_dilation(mask==obj_idx, iterations=4)
		contour = mask==obj_idx - eroded_mask
		mask_contour += contour.astype(mask.dtype)
	return mask_contour

def draw_contour(img:np.array, mask_for_contour:np.array, contour:np.array):
	img[contour > 0] = mask_for_contour[contour>0]

	return img


