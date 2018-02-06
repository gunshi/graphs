import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#visualise individually to debug
#plan how and read sequence of images and their odom
#fusing and checking
#see if fusing acc to centroids is fine


path='/home/gunshi/Downloads/SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_F/000001.png'
synthia_memory_folder = ''
n_memory = 100
memory_odom_list = []
synthia_live_folder = ''
n_live = 10
live_odom_list = []

nets=['enet','GT_synthia']
datasets=['mapillary','synthia','gta']

dataset_to_use = -1
net_to_use = -1

static_synthia = [0,2,3,4,5,6,7,9,11,12,13,14]

synthia_semantic_values = [
	#Void		
	[0,0,0], #!!
	#Sky             
	[128,128,128],
	#Building        
	[128,0,0],
	#Road            
	[128,64,128],
	#Sidewalk        
	[0,0,192],
	#Fence           
	[64,64,128], #!!
	#Vegetation      
	[128,128,0],
	#Pole            
	[192,192,128],
	#Car             
	[64,0,128],
	#Traffic Sign    
	[192,128,128], #!!
	#Pedestrian      
	[64,64,0], #!!
	#Bicycle         
	[0,128,192], #!!
	#Lanemarking	
	[0,175,0], #documentation says 172
	#Traffic Light	
	[0,128,128],
	#correct car
	[156,33,166]	
] 

rgb_mapping = static_synthia

global_graph =  []
global_nodes = []

blob0 = [] 
adj0 = [] 
img0 = [] 
categ0 = []
blob1 = [] 
adj1 = []
img1 = [] 
categ1 = []

def config():
	f=2

def load_odom_data(odom_path):
	#load odom
	fileobj = open(odom_path,'r')
	for line in fileobj:
		words = line.split()
		memory_odom_list.append(words)

def build_graph_synthia():
	img_path = synthia_memory_folder + '%06d.png' % (0,)
	blob1, adj1, img1, categ1 = build_image_graph(img_path)
	empty = [[] for i in range(len(rgb_mapping))]
	global_graph.append(empty)

	#add to global graph
	#make the info list -to go into every semantic place of appropriate frame - centroids, bbox, cnt, stringinfo
	#when fusing, add info for indicating empty/fused distinction?

	add_to_global_graph(0,info_list)

	for i in range(1,n_memory):
		blob0 = blob1
		adj0 = adj1
		img0 = img1
		categ0 = categ1

		img_path = synthia_memory_folder + '%06d.png' % (i,)
		blob1, adj1, img1, categ1 = build_image_graph(img_path)
		global_graph.append(empty)
		#check with previous and add to graph


#fused or not
#fuse info
#edge info

def format_seg(blob_list, categ_list):
	format_list = [[] for i in range(len(rgb_mapping))]

	for cat in categ_list:
		index = categ_list.index(cat)
		for sem_value in rgb_mapping:
			if cat == sem_value:
				index_sem = rgb_mapping.index(sem_value)
				for j in range(len(blob_list[index])):
					blob_list[index][j]['fused'] = False
				format_list[index_sem] = blob_list[index]
				
				break
	return format_list

def check_with_previous_and_fuse(frame_num, ):

	#centroids less than some dist
	#same categ
	#odom

	##yes when fusing, update info at that place 
	##therefore add empty blob with fused = true


def add_to_global_graph(index, info_list): #blobs list acc to semantic categs, with blob infos
	global_graph[index]

def build_seq_graph():
	f=2

def build_image_graph(imgpath):
	img_seg = cv2.imread(imgpath)
	img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
	#for i in range(600,750):
		#print(img_seg[340,i])
	plt.imshow(img_seg)
	plt.show()
	print(img_seg.shape)

	sep_seg_imgs, sep_seg_categs, blob_list, adj_list = get_sep_category_imgs(img_seg)
	sep_denoised = []
	for sep_img in sep_seg_imgs:
		denoised = remove_noise(sep_img)
		sep_denoised.append(denoised)
	blob_list, adj_list = get_blobs_and_centroids(sep_denoised,sep_seg_categs, blob_list, adj_list)
	adj_list = get_edges(sep_denoised, img_seg, blob_list, adj_list)
	return blob_list, adj_list, img_seg, sep_seg_categs

def get_sep_category_imgs(img):
	sep_seg_imgs=[]
	sep_seg_categs=[]
	blob_list = []
	adj_list = []
	for values in rgb_mapping:
		pix_value = synthia_semantic_values[values]
		mask = np.all(img == pix_value, axis=-1)
		if(np.any(mask)):
			sep_seg_imgs.append(mask)
			sep_seg_categs.append(values)
			blob_list.append([])
			adj_list.append([])
			#plt.imshow(mask, cmap='gray')
			#plt.show()
	return sep_seg_imgs, sep_seg_categs, blob_list, adj_list

def get_blobs_and_centroids(sem_imgs,categ_list, blob_list, adj_list):
	iter_img = 0
	for img in sem_imgs:
		print(img)
		index = iter_img
		print(index)
		value = categ_list[index]
		_,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 2)
		img_3channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		conts=[]
		counter = 0
		for cnt in contours:
			if(cv2.contourArea(cnt) > 55):
				id = str(value) + '_' + str(counter)  ##not needed 
				print(cv2.contourArea(cnt))
				conts.append(cnt)
				M = cv2.moments(cnt)
				if(M['m00']!=0):
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])
				else:
					print('is zero')

				x,y,w,h = cv2.boundingRect(cnt)
    			#roi = img[y:y+h,x:x+w]
    			#add this to a list along with coords to compare with others
				blob_info = {'roi':[x,y,w,h],'cnt':cnt,'center':(cx,cy)}
				blob_list[index].append(blob_info)
				adj_list[index].append([])
				counter += 1

		cv2.drawContours(img_3channel, conts, -1, (0,255,0), 3)
		#plt.imshow(img_3channel)
		#plt.show()

		print('................................................................')
		iter_img += 1
		return blob_list, adj_list

def get_edges(sep_denoised, img, blob_list, adj_list):
	imgcopy = img
	assert(len(sep_denoised)==len(blob_list))
	blob_len = len(blob_list)
	for i in range(blob_len):
		for blob_iter_i in range(len(blob_list[i])):
			for j in range(i,blob_len):
				for blob_iter_j in range(len(blob_list[j])):
					x1,y1,w1,h1 = blob_list[i][blob_iter_i]['roi']
					x2,y2,w2,h2 = blob_list[j][blob_iter_j]['roi']
					result = intersection((x1,y1,w1,h1),(x2,y2,w2,h2))
					print(result)
					if(result==()):
						print('yes it is ()')
					else:
						print('no')
						#compute actual intersection
						#cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
						if(w1*h1>w2*h2):
							img_bwa = cv2.bitwise_and(sep_denoised[i][y2:y2+h2,x2:x2+w2],sep_denoised[j][y2:y2+h2,x2:x2+w2])
						else:					
							img_bwa = cv2.bitwise_and(sep_denoised[i][y1:y1+h1,x1:x1+w1],sep_denoised[j][y1:y1+h1,x1:x1+w1])
						if(not np.any(img_bwa)):
							#no intersection
							f=2
						else:
							center1 = blob_list[i][blob_iter_i]['center']
							center2 = blob_list[j][blob_iter_j]['center']
							adj_list[i][blob_iter_i] = (j,blob_iter_j)
							adj_list[j][blob_iter_j] = (i,blob_iter_i)
							#draw edges and centers in image for display
							cv2.line(imgcopy, center1, center2, (0,0,0), 2)

	plt.imshow(imgcopy)
	plt.show()
	return adj_list

def remove_noise(img):	
	kernel = np.ones((9,9),np.uint8)
	#erosion = cv2.erode(img,kernel,iterations = 1)
	img = np.array(img * 255, dtype = np.uint8)
	#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	#plt.imshow(opening, cmap='gray')
	#plt.show()
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
	closing = cv2.dilate(closing,kernel,iterations = 1)
	#plt.imshow(closing, cmap='gray')
	#plt.show()
	return closing

def get_descriptors(graph):
	desc=[]
	return desc

def union(a,b):
  	x = min(a[0], b[0])
  	y = min(a[1], b[1])
  	w = max(a[0]+a[2], b[0]+b[2]) - x
  	h = max(a[1]+a[3], b[1]+b[3]) - y
  	return (x, y, w, h)

def intersection(a,b):
  	x = max(a[0], b[0])
  	y = max(a[1], b[1])
  	w = min(a[0]+a[2], b[0]+b[2]) - x
  	h = min(a[1]+a[3], b[1]+b[3]) - y
  	if w<0 or h<0: return () # or (0,0,0,0) ?
  	return (x, y, w, h)

build_image_graph(path)