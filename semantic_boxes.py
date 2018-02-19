import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from transformations import euler_from_matrix
from random import random
import time

static_dict = [0,3,5,6,7,8,9]
semantic_dict = [
	[64, 64, 0],
	[0, 0, 128], ##no
	[0, 0, 0],  ##no
	[0, 128, 128],
	[128, 128, 128], ##no
	[64, 0, 192],
	[128, 0, 64],
	[0, 192, 64],
	[128, 64, 128],
	[192, 0, 0] ]


sem_path = '/home/gunshi/Downloads/labelling_release/test/gt/'
img_path = '/home/gunshi/Downloads/labelling_release/test/images/'


def get_unique_colors(img):
	return set( tuple(v) for m2d in img for v in m2d )

def get_sep_category_imgs(img):
	sep_seg_imgs=[]
	for values in static_dict:
		pix_value = semantic_dict[values]
		mask = np.all(img == pix_value, axis=-1)
		if(np.any(mask)):
			sep_seg_imgs.append(mask)
			plt.imshow(mask, cmap='gray')
			plt.show()
		else:
			sep_seg_imgs.append(None)
	return sep_seg_imgs

def remove_noise(img):
	kernel = np.ones((9,9),np.uint8)
	img = np.array(img * 255, dtype = np.uint8)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
	closing = cv2.dilate(closing,kernel,iterations = 1)
	#plt.imshow(closing, cmap='gray')
	#plt.show()
	return closing

def get_blobs_and_boxes(sem_imgs, real_img):
	height, width, channels = real_img.shape
	categ_wise_boxes = [[] for i in range(len(static_dict))]

	counter = 0
	for img in sem_imgs:
		if(img==None):
			counter+=1
			continue
		_,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 2)
		#img_3channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		for cnt in contours:
			cnt_area = cv2.contourArea(cnt)
			if(cnt_area > 2000):
				print(cnt_area)
				"""
				conts.append(cnt)
				M = cv2.moments(cnt)
				if(M['m00']!=0):
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])
				"""
				x,y,w,h = cv2.boundingRect(cnt)
				roi = real_img[max(0,y-20):min(height,y+h+20),max(0,x-20):min(width,x+w+20)]
				categ_wise_boxes[counter].append(roi)
				plt.imshow(roi)
				plt.show()
		counter +=1


	return categ_wise_boxes


def build_image_graph():

	for i in range(1,20):
		# img_seg_path = sem_path + str(i) + '.png'
		# img_seg = cv2.imread(img_seg_path)
		img_seg = cv2.imread("../i1.png")
		#img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
		# print(img_seg.shape)
		# assert(1==2)
		colors = get_unique_colors(img_seg)
		# print(colors)
		# assert(1==2)
		# img_path = "../"
		# real_img_path = img_path + str(i) + '.png'
		real_img_path = "../i1.png"

		real_img = cv2.imread(real_img_path)
		# print(real_img.shape)
		# assert(1==2)

		sep_seg_imgs = get_sep_category_imgs(img_seg) #one entry per static semantic category

		# assert(1==2)

		sep_denoised = []
		for sep_img in sep_seg_imgs:
			if(sep_img==None):
				sep_denoised.append(None)
				continue
			else:
				denoised = remove_noise(sep_img)
				sep_denoised.append(denoised)
		categ_wise_boxes = get_blobs_and_boxes(sep_denoised,real_img)
		#compute embeddings

	return img_seg

#maintain scale
a = build_image_graph()
plt.imshow(a)
