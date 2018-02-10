import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from transformations import euler_from_matrix
from random import random

#need to do query graph filtering also
#live + memory both case handling
#how to fuse cross image blobs
#adjacency list undirectedness, all edges covered
#random walk
#check why they're not fused with the right blobs///////////////////////////////////////
#matching and selection of descriptors
#get stats on most and least occuring blobs
#need some 3d merging also

path='/home/gunshi/Downloads/SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_F/000001.png'
synthia_memory_folder = '/home/gunshi/Downloads/SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_F/'
memory_odom_path = '/home/gunshi/Downloads/SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Left/Omni_F/concat.txt'
n_memory = 100
memory_odom_list = []
memory_odom_list_filtered = []
frames_memory_filtered = []

synthia_live_folder = ''
n_live = 10
live_odom_list = []
frames_live =[]

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
random_walk_desc = []

n_desc = 150
walk_length = 3

blob0 = [] 
adj0 = [] 
img0 = [] 
categ0 = []
blob1 = [] 
adj1 = []
img1 = [] 
categ1 = []

thresh = 100
area_thresh = 1.3
thresh_fused = 0
area_thresh_fused = 0


def compute_rel_odom(matA, matB): #A is src, B is dest
    matA_inv=np.linalg.inv(matA)
    #src inv * tgt = relative transform
    rel_odom_src_tgt=np.matmul(matA_inv,matB)
    ##converting to euler to get 6d =(3+3) dimensional vector for pose
    rel_tform_rot=rel_odom_src_tgt[0:3,0:3]
    rx,ry,rz = euler_from_matrix(rel_tform_rot)
    rel_tform_vec = [ rel_odom_src_tgt[0,3], rel_odom_src_tgt[1,3], rel_odom_src_tgt[2,3], rx, ry, rz]
    a = np.array((rel_odom_src_tgt[0,3], rel_odom_src_tgt[1,3], rel_odom_src_tgt[2,3]))
    dist = np.linalg.norm(a)
    return rel_tform_vec, dist

def load_odom_data(odom_path, memory = 0):
	#load odom
	fileobj = open(odom_path,'r')
	for line in fileobj:
		words = line.split()
		nums = np.array([float(ele) for ele in words])
		nums = np.reshape(nums, (4,4), order='F')
		if memory:
			memory_odom_list.append(nums)
		else:
			live_odom_list.append(nums)

def get_live_odom_subset(odom_list, list_range):
	return 

def build_graph_synthia(odom_list):
	img_path = synthia_memory_folder + '%06d.png' % (0,)
	blob1, adj1, img1, categ1, _ = build_image_graph(img_path)
	empty = [[] for i in range(len(rgb_mapping))]
	global_graph.append(empty)
	random_walk_desc.append(empty)
	info_list = format_seg( blob1, categ1, adj1, 0, 0) #0 indexed?
	add_to_global_graph(0,0,info_list)
	origin = odom_list[0]
	frames_memory_filtered.append(0)
	counter = 1

	for i in range(1,n_memory):
		dest = odom_list[i]
		#decide acc to odom and skip
		rel_odom, dist = compute_rel_odom(origin, dest)
		print('dist of frame '+str(i)+' = '+str(dist))
		if(dist < 0.03):
			print('skipping')
			#skip this frame
			continue
		else:
			origin = dest
			print('frame '+str(i))
			frames_memory_filtered.append(i)

			blob0 = blob1
			adj0 = adj1
			img0 = img1
			categ0 = categ1

			img_path = synthia_memory_folder + '%06d.png' % (i,)
			blob1, adj1, img1, categ1, imgcurrent = build_image_graph(img_path)
			global_graph.append(empty)
			random_walk_desc.append(empty)

			info_list = format_seg(blob1, categ1, adj1, i, counter )
			info_list = check_with_previous_and_fuse(i, counter, info_list, imgcurrent)
			add_to_global_graph(i, counter, info_list)
			counter += 1

		#check with previous and add to graph


def format_seg(blob_list, categ_list, adj_list, frame_num, position_num):
	format_list = [[] for i in range(len(rgb_mapping))]

	for cat in categ_list:
		index = categ_list.index(cat)
		for sem_value in rgb_mapping:
			if cat == sem_value:
				index_sem = rgb_mapping.index(sem_value)
				for j in range(len(blob_list[index])):
					blob_list[index][j]['fused'] = False
					blob_list[index][j]['localParent'] = []
					blob_list[index][j]['localParentOf'] = []
					blob_list[index][j]['fuseInfo'] = []
					blob_list[index][j]['frameNum'] = frame_num
					blob_list[index][j]['parentOf'] = []
					blob_list[index][j]['fuseDepth'] = 0

					edge_list = adj_list[index][j]  ##################################
					edge_list_formatted = []
					#[0] is index into adjlist, [1] is blob position
					for edge_ele in edge_list:
						index_to_convert = edge_ele[0]
						index_converted = rgb_mapping.index(categ_list[index_to_convert])
						edge_list_formatted.append((frame_num, position_num, index_converted,edge_ele[1]))
					blob_list[index][j]['edgeInfo'] = edge_list_formatted


				format_list[index_sem] = blob_list[index]
				
				break
	return format_list



def check_with_previous_and_fuse(frame_num, position_num, info_list, imgcurrent):
	print('PREVIOUS AND FUSE')
	print(frame_num)
	print(position_num)
	#centroids less than some dist
	#same categ
	#odom-----
	#area needs to be around the same(slightly larger)
	#bounding box/convex hull roughly the same-----
	#contour overlap?

	ref_list = global_graph[position_num-1]
	print('len ref list')
	print(len(ref_list))
	for sem_categ in range(len(rgb_mapping)):
		
		if(info_list[sem_categ] and ref_list[sem_categ]):
			print('not denied')
			counter= 0 
			for blob in info_list[sem_categ]:
				counter_ref = 0
				for blob_ref in ref_list[sem_categ]:
					dist = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
					area_ratio = blob['area']/(1.*blob_ref['area']) ##is this float division?
					#print('dist of centroid' +str(dist))
					#print('area ratio '+str(area_ratio))
					if(dist < thresh and area_ratio < area_thresh and area_ratio > 0.9):
						print('FUSING')
						#add condition to see if not already fused
						if(blob_ref['fused']):
						# assign global and local parents
							fuse_info = blob_ref['fuseInfo']
							global_graph[fuse_info[0]][fuse_info[1]][fuse_info[2]]['parentOf'].append((position_num, sem_categ, counter)) #####  change
							#can optionally add local child info to local parent also 
							#
							blob_ref['localParentOf'].append((position_num, sem_categ, counter))
							blob['fuseDepth'] = blob_ref['fuseDepth']+1
							blob['fused'] =  True
							blob['fuseInfo'] = fuse_info
							blob['localParent'] = [position_num-1, sem_categ, counter_ref]
						else:
							blob_ref['localParentOf'].append((position_num, sem_categ, counter))
							blob_ref['parentOf'].append((position_num, sem_categ, counter))
							blob['fused'] =  True
							blob['fuseInfo'] = [position_num-1, sem_categ, counter_ref]
							blob['localParent'] = [position_num-1, sem_categ, counter_ref]
							blob['fuseDepth'] = 1
						#cv2.line(imgcurrent, blob['center'], blob_ref['center'], (255,255,255), 2)
						cv2.circle(imgcurrent,blob['center'], 8, (255,255,255), 2)
					counter_ref += 1
				counter +=1
			#recovery tactic here
			counter_ref = 0
			counter = 0

			for blob_ref in ref_list[sem_categ]:
				if(not blob_ref['parentOf']):
					for blob in info_list[sem_categ]:
						if(blob['fused']):
							dist = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
							area_ratio = blob['area']/(1.*blob_ref['area'])

							if(dist < thresh and area_ratio < area_thresh and area_ratio > 0.9):
								print('CHANGING')
								#unset previous parent's info!
								assert(blob['fused'])
								parent_tuple = blob['localParent']
								parent_blob = global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]
								if(len(parent_blob['localParentOf'])>2):

									if(parent_tuple == blob['fuseInfo']):
										to_remove = global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['parentOf'].index((position_num, sem_categ, counter))
										to_remove2 = global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'].index((position_num, sem_categ, counter))

										del global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['parentOf'][to_remove]
										del global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'][to_remove2]

										blob_ref['parentOf'].append((position_num, sem_categ, counter))
										blob_ref['localParentOf'].append((position_num, sem_categ, counter))
										blob['fused'] =  True
										blob['fuseInfo'] = [position_num-1, sem_categ, counter_ref]
										blob['localParent'] = [position_num-1, sem_categ, counter_ref]

									else:
										global_par = blob['fuseInfo'] #global parent
										
										to_remove = global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['parentOf'].index((position_num, sem_categ, counter))
										to_remove2 = global_graph[global_par[0]][sem_categ][global_par[2]]['localParentOf'].index((position_num, sem_categ, counter))

										del global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['parentOf'][to_remove]
										del global_graph[global_par[0]][sem_categ][global_par[2]]['localParentOf'][to_remove2]



##change
										blob_ref['parentOf'].append((position_num, sem_categ, counter))
										blob_ref['localParentOf'].append((position_num, sem_categ, counter))
										blob['fused'] =  True
										blob['fuseInfo'] = [position_num-1, sem_categ, counter_ref]
										blob['localParent'] = [position_num-1, sem_categ, counter_ref]



						counter +=1
				counter_ref += 1


				#draw lines between centres and disp image
				for blob in info_list[sem_categ]:
					if blob['fused']:
						parent_tuple = blob['fuseInfo']
						#print(parent_tuple)
						cv2.line(imgcurrent, blob['center'],global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['center'], (255,255,255), 2)




		else:
			print('if denied')
	#disp image
	print('displaying fused info image')
	plt.imshow(imgcurrent)
	plt.show()
	return info_list

						


def add_to_global_graph(frame_num, position_num, blob_list): #blobs list acc to semantic categs, with blob infos
	global_graph[position_num] = blob_list
	print(len(blob_list))
	print('add to global graph at '+str(position_num))
	print(len(global_graph))
	print(len(global_graph[position_num]))
	print('...........')


def random_walk(walk_left, walk_length, list_not, position_num, sem_categ, blob_num):
	#list_not should contain 
	#-sem_categ
	#-frame num
	#-blob num

	f=2 
	"""
	if (walk_left==0):
		list_not.append((position_num,sem_categ,blob_num))
		head = list_not[0]
		for el in list_not:
			random_walk_desc[head[0]][head[1]][head[2]].append((el[0],el[1],el[2]))
		return
	no_neigh = True
	global_len = len(global_graph)
	blob = global_graph[position_num][sem_categ][blob_num]
	if(not blob['fused']):
		if(blob['parentOf']):
			cutoff = 0.5
		else:
			cutoff= 1

		if (random() < cutoff):
			#find same image neighbours
			#what if neighbour is fused
			edge_list = blob['edgeInfo']
			for edge in edge_list:
				(frame_num, position_num, sem, blob_pos) = edge
				if():
				list_not.append([])
				random_walk(walk_left-1, walk_length, list_not, )
		else:
			#find fused blob's neighbours
			fuse_list = blob['parentOf']
			for children_info in fuse_list:
				child_blob = global_graph[children_info[0]][children_info[1]][children_info[2]]
				list_not.append([])
				random_walk(walk_left-1, walk_length, list_not,)

	if(no_neigh):
		print('NO NEIGHBOURS')
		return 
	"""
def find_neighbour():
	f=2
def compute_matching(pos1, sem1, blob1, pos2, sem2, blob2):
	f=2

	#will we ever match for diff sems?

	desc1 = random_walk_desc[pos1][sem1][blob1]
	desc2 = random_walk_desc[pos2][sem2][blob2]
	#make np arrays of semantics out of this

	sub = np.subtract(desc_1,desc_2)
	sub_size = len(sub)
	zero_els = np.count_nonzero(sub==0) 

	#make str if want edit dest

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
def call_random_walk():
	for blob_list in global_graph:
		for sem_cat in range(len(rgb_mapping)):
			counter = 0
			for blob in blob_list[sem_cat]:
				random_walk(walk_length, walk_length, [], sem_cat, counter)
				counter +=1

def build_image_graph(imgpath):
	img_seg = cv2.imread(imgpath)
	img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)

	#for i in range(600,750):
		#print(img_seg[340,i])
	#(img_seg)
	#plt.show()
	print(img_seg.shape)

	sep_seg_imgs, sep_seg_categs, blob_list, adj_list = get_sep_category_imgs(img_seg)
	sep_denoised = []
	for sep_img in sep_seg_imgs:
		denoised = remove_noise(sep_img)
		sep_denoised.append(denoised)
	blob_list, adj_list = get_blobs_and_centroids(sep_denoised,sep_seg_categs, blob_list, adj_list)
	adj_list = get_edges(sep_denoised, img_seg, blob_list, adj_list)
	return blob_list, adj_list, img_seg, sep_seg_categs, img_seg 

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
		#print(img)
		index = iter_img
		#print(index)
		value = categ_list[index]
		_,contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 2)
		img_3channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		conts=[]
		counter = 0
		for cnt in contours:
			cnt_area = cv2.contourArea(cnt)
			if(cnt_area > 55):
				id = str(value) + '_' + str(counter)  ##not needed 
				#print(cnt_area)
				conts.append(cnt)
				M = cv2.moments(cnt)
				if(M['m00']!=0):
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])

				x,y,w,h = cv2.boundingRect(cnt)
    			#roi = img[y:y+h,x:x+w]
    			#add this to a list along with coords to compare with others
				blob_info = {'roi':[x,y,w,h],'cnt':cnt,'center':(cx,cy), 'area':cnt_area}
				#print('index '+str(index))
				#print('appending')
				blob_list[index].append(blob_info)
				adj_list[index].append([])
				counter += 1

		#cv2.drawContours(img_3channel, conts, -1, (0,255,0), 3)
		#plt.imshow(img_3channel)
		#plt.show()

		#print('................................................................')
		iter_img += 1
	return blob_list, adj_list

def get_edges(sep_denoised, img, blob_list, adj_list):
	imgcopy = img
	assert(len(sep_denoised)==len(blob_list))
	blob_len = len(blob_list)
	#print('blob len')
	#print(blob_len)
	for i in range(blob_len):
		#print(i)
		for blob_iter_i in range(len(blob_list[i])):
			#print('uppest iter')
			for j in range(i+1,blob_len):
				#print('i,j = '+str(i)+'....'+str(j))
				#print('upper iter')
				for blob_iter_j in range(len(blob_list[j])):
					#print('one iter')
					x1,y1,w1,h1 = blob_list[i][blob_iter_i]['roi']
					x2,y2,w2,h2 = blob_list[j][blob_iter_j]['roi']
					result = intersection((x1,y1,w1,h1),(x2,y2,w2,h2))
					#print(result)
					if(result!=()):
						#print('yes it is ()')
						#print('no')
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
							#print('there is overlap')
							center1 = blob_list[i][blob_iter_i]['center']
							center2 = blob_list[j][blob_iter_j]['center']
							#print(center1)
							#print(center2)

							adj_list[i][blob_iter_i].append((j,blob_iter_j))
							adj_list[j][blob_iter_j].append((i,blob_iter_i))
							#draw edges and centers in image for display
							cv2.circle(imgcopy,center1, 3, (255,255,255), -1)
							cv2.circle(imgcopy,center2, 3, (255,255,255), -1)

							cv2.line(imgcopy, center1, center2, (0,0,0), 2)

							#print('drew line')

	#plt.imshow(imgcopy)
	#plt.show()
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

#build_image_graph(path)
load_odom_data(memory_odom_path,True)
build_graph_synthia(memory_odom_list)