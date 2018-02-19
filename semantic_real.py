import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from transformations import euler_from_matrix
from random import random																		
import time
from collections import Counter


'''
Classes:
------------
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]

Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
'''
static_enet=[0,1,2,3,4,5,6,7]

label_to_colours = [
	#0: 
	[128,128,128],
	#1: 
	[128,0,0],
	#2: 
	[192,192,128],
	#3: 
	[128,64,128],
	#4: 
	[60,40,222],
	#5: 
	[128,128,0],
	#6: 
	[192,128,128],
	#7: 
	[64,64,128],

	#8: 
	[64,0,128],
	#9: 
	[64,64,0],
	#10: 
	[0,128,192],
	#11: 
	[0,0,0]
]


synthia_memory_folder_rgb = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Left/Omni_F/'
synthia_live_folder_rgb = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-NIGHT/RGB/Stereo_Left/Omni_F/'


path='/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_F/000001.png'
synthia_memory_folder = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-DAWN/GT/COLOR/Stereo_Left/Omni_F/'
memory_odom_path = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-DAWN/CameraParams/Stereo_Left/Omni_F/concat.txt'
synthia_memory_folder_depth = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-DAWN/Depth/Stereo_Left/Omni_F/'


live_odom_path = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-NIGHT/CameraParams/Stereo_Left/Omni_B/concat.txt'

n_memory = 250
memory_odom_list = []
memory_odom_list_filtered = []

frames_memory_filtered = []
frames_live_filtered = []

synthia_live_folder_depth = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-NIGHT/Depth/Stereo_Left/Omni_B/'
synthia_live_folder = '/home/gunshi/Downloads/synthia/SYNTHIA-SEQS-01-NIGHT/GT/COLOR/Stereo_Left/Omni_B/'
n_live = 100
live_odom_list = []
frames_live =[]


max_fuse_depth = 0

nets=['enet','GT_synthia']
datasets=['mapillary','synthia','gta']

dataset_to_use = -1
net_to_use = -1

static_synthia = [0,2,4,5,6,7,9,12,13]

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

rgb_mapping = static_enet #static_synthia

global_graph =  []
random_walk_desc = []

global_graph_live =  []
random_walk_desc_live = []
random_walk_sems_memory = []
random_walk_sems_live = []

n_desc = 85
walk_length = 3

"""
blob0 = [] 
adj0 = [] 
img0 = [] 
categ0 = []
blob1 = [] 
adj1 = []
img1 = [] 
categ1 = []
"""

thresh = 80
area_thresh = 1.25
thresh_fused = 70
area_thresh_fused = 1.2
matches = []


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

def build_graph_synthia(odom_list):
	img_path = synthia_memory_folder + '%06d.png' % (0,)
	blob1, adj1, img1, categ1, _ = build_image_graph(img_path)
	empty = [[] for i in range(len(rgb_mapping))]
	global_graph.append(empty)
	random_walk_desc.append(empty)
	random_walk_sems_memory.append(empty)
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
		if(dist < 0.1):
			print('skipping')
			#skip this frame
			continue
		else:
			if(dist<0.2):
				thresh = 82
			else:
				thresh = 90
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
			random_walk_sems_memory.append(empty)
			info_list = format_seg(blob1, categ1, adj1, i, counter )
			info_list = check_with_previous_and_fuse(i, counter, info_list, imgcurrent)
			add_to_global_graph(i, counter, info_list)
			counter += 1

	call_random_walk()
	see_desc(0,0,0)

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
						#print(index_to_convert)
						#print(index_converted)
						#print('...')
						#time.sleep(1)
						edge_list_formatted.append((frame_num, position_num, index_converted,edge_ele[1]))
					#print(edge_list_formatted)
					blob_list[index][j]['edgeInfo'] = edge_list_formatted
					#time.sleep(2)

				format_list[index_sem] = blob_list[index]
				
				break
	return format_list

def check_with_previous_and_fuse(frame_num, position_num, info_list, imgcurrent):
	imgcurrent2 = np.copy(imgcurrent)
	#print('PREVIOUS AND FUSE')
	#print(frame_num)
	#print(position_num)
	#centroids less than some dist
	#same categ
	#odom-----
	#area needs to be around the same(slightly larger)
	#bounding box/convex hull roughly the same-----
	#contour overlap?

	ref_list = global_graph[position_num-1]
	#print('len ref list')
	#print(len(ref_list))
	for sem_categ in range(len(rgb_mapping)):
		
		if(info_list[sem_categ] and ref_list[sem_categ]):
			#print('not denied')
			counter= 0 
			for blob in info_list[sem_categ]:
				counter_ref = 0
				for blob_ref in ref_list[sem_categ]:
					dist = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
					area_ratio = blob['area']/(1.*blob_ref['area']) ##is this float division?
					#print('dist of centroid' +str(dist))
					#print('area ratio '+str(area_ratio))
					if(dist < thresh and area_ratio < area_thresh and area_ratio > 0.78):
						"""
						if(sem_categ==0):
							cv2.drawContours(imgcurrent2, [blob['cnt']], 0, (255,255,255), 6)
							cv2.drawContours(imgcurrent2, [blob_ref['cnt']], 0, (0,255,0), 6) ####
							plt.imshow(imgcurrent2)
							plt.show()
						"""
						#if(blob_ref['parentOf']):
							#print('FUSING TO ONE THAT IS ALREADY A PARENT')
							#print('dist of centroid' +str(dist))
							#print('area ratio '+str(area_ratio))
						#add condition to see if not already fused
						if(blob_ref['fused']):
							if(blob_ref['fuseDepth']>6):
								counter_ref+=1
								continue
							#print('GLOBAL AND LOCAL')
						# assign global and local parents
							print('FUSING')

							fuse_info = blob_ref['fuseInfo']
							global_graph[fuse_info[0]][fuse_info[1]][fuse_info[2]]['parentOf'].append((position_num, sem_categ, counter)) #####  change
							#can optionally add local child info to local parent also 
							#
							blob_ref['localParentOf'].append((position_num, sem_categ, counter))
							blob['fuseDepth'] = blob_ref['fuseDepth']+1
							global max_fuse_depth
							if(blob['fuseDepth']>max_fuse_depth):
								max_fuse_depth = blob['fuseDepth']
							blob['fused'] =  True
							blob['fuseInfo'] = fuse_info
							blob['localParent'] = [position_num-1, sem_categ, counter_ref]
						else:
							print('FUSING')
							blob_ref['localParentOf'].append((position_num, sem_categ, counter))
							blob_ref['parentOf'].append((position_num, sem_categ, counter))
							blob['fused'] =  True
							blob['fuseInfo'] = [position_num-1, sem_categ, counter_ref]
							blob['localParent'] = [position_num-1, sem_categ, counter_ref]
							blob['fuseDepth'] = 1
						#cv2.line(imgcurrent, blob['center'], blob_ref['center'], (255,255,255), 2)
						#length = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
						#print('early line length is: '+str(length))
						cv2.circle(imgcurrent,blob['center'], 8, (255,255,255), 2)
						break
					counter_ref += 1
				counter +=1
			#recovery tactic here
			counter_ref = 0



			for blob_ref in ref_list[sem_categ]:
				if(not blob_ref['parentOf']):
					counter = 0
					for blob in info_list[sem_categ]:
						if(blob['fused']):
							dist = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
							area_ratio = blob['area']/(1.*blob_ref['area'])
							#if(sem_categ==0 and blob_ref['center'][0]<550 and blob_ref['center'][1]<550):
								#imgcurrent3 = np.copy(imgcurrent2)
								#cv2.drawContours(imgcurrent3, [blob_ref['cnt']], 0, (255,255,255), 6) ####
								#cv2.drawContours(imgcurrent3, [blob['cnt']], 0, (0,0,255), 6) ####
								#plt.imshow(imgcurrent3)
								#plt.show()

							if(dist < thresh and area_ratio < area_thresh and area_ratio > 0.78):
								#print('CHANGING')

								#unset previous parent's info!
								assert(blob['fused'])
								parent_tuple = blob['localParent']
								parent_blob = global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]


								if(len(parent_blob['localParentOf'])>2):
									"""
									if(sem_categ==0):
										print(dist)
										print(area_ratio)
										cv2.drawContours(imgcurrent, [blob['cnt']], 0, (255,255,255), 6)
										cv2.drawContours(imgcurrent, [blob_ref['cnt']], 0, (0,255,0), 6) ####
										cv2.drawContours(imgcurrent, [parent_blob['cnt']], 0, (0,0,255), 6)
										plt.imshow(imgcurrent)
										plt.show()
									"""
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
										
										to_remove = global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'].index((position_num, sem_categ, counter))
										to_remove2 = global_graph[global_par[0]][sem_categ][global_par[2]]['parentOf'].index((position_num, sem_categ, counter))

										del global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'][to_remove]
										del global_graph[global_par[0]][sem_categ][global_par[2]]['parentOf'][to_remove2]

										if(blob_ref['fused']):
										# assign global and local parents
											fuse_info = blob_ref['fuseInfo']
											global_graph[fuse_info[0]][fuse_info[1]][fuse_info[2]]['parentOf'].append((position_num, sem_categ, counter))						
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


						counter +=1
				counter_ref += 1


				#draw lines between centres and disp image
				for blob in info_list[sem_categ]:
					if blob['fused']:
						parent_tuple = blob['fuseInfo']
						#print(parent_tuple)
						cv2.line(imgcurrent, blob['center'],global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['center'], (255,255,255), 2)
						length = np.linalg.norm(np.array(blob['center'])-np.array(global_graph[parent_tuple[0]][sem_categ][parent_tuple[2]]['center']))
						#if(sem_categ==0 and blob['center'][0]<550 and blob['center'][1]<550):
							#print('line length is: '+str(length))



		#else:
		#	print('if denied')
	#disp image
	#print('displaying fused info image')
	#plt.imshow(imgcurrent)
	#plt.show()
	return info_list					

def add_to_global_graph(frame_num, position_num, blob_list): #blobs list acc to semantic categs, with blob infos
	global_graph[position_num] = blob_list
	random_walk_desc[position_num] = [[[] for blob in semcatblobs] for semcatblobs in blob_list]
	random_walk_sems_memory[position_num] = [[[] for blob in semcatblobs] for semcatblobs in blob_list]
	print(len(blob_list))
	print('add to global graph at '+str(position_num))
	print(len(global_graph))
	print(len(global_graph[position_num]))
	print('...........')

def random_walk(walk_left, walk_length, list_not, position_num, sem_categ, blob_num, fused_case = False):

	"""
	if(len(list_not)>=4):
		print('----------------')
		print(list_not)
		print(walk_left)
		print(walk_length)
		print('last node '+str(position_num)+','+str(sem_categ)+','+str(blob_num)+' of list')

		print('----------------')
		time.sleep(20)
	"""
	#print('walk left '+str(walk_left))
	if(fused_case):
		assert(walk_left!=0)
	if (walk_left==0):
		list_not.append((position_num,sem_categ,blob_num))
		#print(list_not)
		#time.sleep(1)
		head = list_not[0]
		"""
		print('----------------')
		print('last list')
		print(list_not)
		print('----------------')
		#print(len(list_not))
		"""
		#print('walk ended '+str(head[0]) + '..' + str(head[1]) + '..' + str(head[2]))
		random_walk_desc[head[0]][head[1]][head[2]].append([])
		random_walk_sems_memory[head[0]][head[1]][head[2]].append([])

		for el in list_not:
			random_walk_desc[head[0]][head[1]][head[2]][-1].append((el[0],el[1],el[2]))
			random_walk_sems_memory[head[0]][head[1]][head[2]][-1].append(el[1])
		random_walk_sems_memory[head[0]][head[1]][head[2]][-1] = tuple(random_walk_sems_memory[head[0]][head[1]][head[2]][-1])
		return

	no_neigh = True


	blob = global_graph[position_num][sem_categ][blob_num]
	edge_list = blob['edgeInfo']

	for edge in edge_list:

		(frame_num, position, sem, blob_pos) = edge
		if([position, sem, blob_pos] not in list_not):
			edge_blob = global_graph[position][sem][blob_pos]
			if(edge_blob['fused']):
				continue
			no_neigh = False
			if(fused_case):
				list_not_temp = list_not[:]
				"""
				print('--------------------------')
				print('node is a fused child case')
				print('walk left '+str(walk_left))
				print('checking out '+str(position)+','+str(sem)+','+str(blob_pos))
				print(list_not_temp)

				print('--------------------------')
				"""
				random_walk(walk_left-1, walk_length, list_not_temp, position, sem, blob_pos, False) ## minus one?


			else:	
				list_not2 = list_not[:]
				list_not2.append([position_num,sem_categ,blob_num])
				"""
				print('--------------------------')
				print('walk left '+str(walk_left))
				print('appending '+str(position_num)+','+str(sem_categ)+','+str(blob_num)+' to the list')
				print('checking out '+str(position)+','+str(sem)+','+str(blob_pos))
				print(list_not2)

				print('--------------------------')
				"""
				random_walk(walk_left-1, walk_length, list_not2, position, sem, blob_pos, False)

	if(not fused_case):
		fuse_list = blob['parentOf']
		for children_info in fuse_list:
			no_neigh = False
			#child_blob = global_graph[children_info[0]][children_info[1]][children_info[2]]
			list_not2 = list_not[:]
			list_not2.append([position_num,sem_categ,blob_num])
			"""
			print('--------------------------')
			print('going to fused child')

			print('walk left '+str(walk_left))
			print('appending '+str(position_num)+','+str(sem_categ)+','+str(blob_num)+' to the list')
			print('checking out child '+str(children_info[0])+','+str(children_info[1])+','+str(children_info[2]))
			print(list_not2)
			print('--------------------------')
			"""
			random_walk(walk_left, walk_length, list_not2, children_info[0],children_info[1],children_info[2], True)

	if(no_neigh):
		print('NO NEIGHBOURS')
		return 

def compute_matching(pos1, sem1, blob1, pos2, sem2, blob2):

	#will we ever match for diff sems?

	desc1 = random_walk_sems_memory[pos1][sem1][blob1]
	desc2 = random_walk_sems_live[pos2][sem2][blob2]
	#make np arrays of semantics out of this
	c1 = Counter(desc1)
	c2 = Counter(desc2)
	intersect = c1 & c2
	print('len of intersect')
	print (len(intersect))
	if(len(intersect)==0):
		#print(c1)
		#print(c2)
		return
	sum = 0
	for key in intersect:
		sum += intersect[key]
	matches.append((sum, len(intersect),pos1,pos2, sem1))
	print('sum')
	print(sum)

	#sub = np.subtract(desc_1,desc_2)
	#sub_size = len(sub)
	#zero_els = np.count_nonzero(sub==0) 

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
	counter_frame = 0
	for frame_list in global_graph:
		for sem_cat in range(len(rgb_mapping)):
			counter = 0
			for blob in frame_list[sem_cat]:
				if(not blob['fused']):
					print('calling for frame' + str(counter_frame))
					print('semcat '+str(sem_cat))
					random_walk(walk_length, walk_length, [], counter_frame, sem_cat, counter, False) ##do we need to pass here?
				counter +=1
		counter_frame +=1
	print(len(random_walk_desc[0][0][0]))
	print(len(random_walk_desc[1][0][0]))
	print(len(random_walk_desc[2][0][0]))
	print(len(random_walk_desc[3][0][0]))
	#print(len(random_walk_desc[4][0][0]))
	#print(len(random_walk_desc[5][0][0]))

	blob = global_graph[0][0][0]
	cnt = blob['cnt']
	blank = np.zeros((760,1280,3), np.uint8)
	cv2.drawContours(blank, [cnt], 0, (255,255,255), 3)
	plt.imshow(blank)
	plt.show()
	for t in random_walk_desc[0][0][0]:
		print(t)
	print(len(random_walk_desc[0][0][0]))
	print(len(list(set(random_walk_sems_memory[0][0][0]))))
	print(set(random_walk_sems_memory[0][0][0]))
	print(Counter(random_walk_sems_memory[0][0][0]))
def see_desc(ind1,ind2,ind3):
	blank_image = np.zeros((1900,300,3), np.uint8)
	blank_image[:,:] = (255,255,255)      # (B, G, R)
	x=0
	for desc in random_walk_desc[ind1][ind2][ind3][:200]:
		y=0
		#change rows
		for node in desc:
			#keep shifting columns
			#print('node')
			#print(node[1])
			print(synthia_semantic_values[rgb_mapping[node[1]]])
			blank_image[x:x+8,y:y+26] = synthia_semantic_values[rgb_mapping[node[1]]]
			y += 26

		x +=8
	#blank_image = cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)
	plt.imshow(blank_image)
	plt.show()


def build_image_graph(imgpath):
	print(imgpath)
	img_seg = cv2.imread(imgpath)
	print(img_seg.shape)
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
			#if(values==0):
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
			if(cnt_area > 200):
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



#=======================================================================================================================

## for live graph=======================================================================================================

#call both live and memory
#match with memory--
#what is that brown
#how to subselect desc

def build_live_graph_synthia(odom_list, start, end):
	img_path = synthia_live_folder + '%06d.png' % (start,)
	depth_img_path = synthia_live_folder_depth + '%06d.png' % (start,)


	blob1, adj1, img1, categ1, segimg = build_image_graph(img_path)
	empty = [[] for i in range(len(rgb_mapping))]
	global_graph_live.append(empty)
	random_walk_desc_live.append(empty)
	random_walk_sems_live.append(empty)
	info_list = format_seg( blob1, categ1, adj1, start, 0) 
	add_to_global_graph_live(start,0,info_list)
	origin = odom_list[start]
	frames_live_filtered.append(start)
	counter = 1

	for i in range(start+1,end):
		dest = odom_list[i]
		#decide acc to odom and skip
		rel_odom, dist = compute_rel_odom(origin, dest)
		print('dist of frame '+str(i)+' = '+str(dist))
		if(dist < 0.1):
			print('skipping')
			#skip this frame
			continue
		else:
			if(dist<0.2):
				thresh = 82
			else:
				thresh = 95
			origin = dest
			print('frame '+str(i))
			frames_live_filtered.append(i)

			blob0 = blob1
			adj0 = adj1
			img0 = img1
			categ0 = categ1

			img_path = synthia_live_folder + '%06d.png' % (i,)
			img_path_depth = synthia_live_folder + '%06d.png' % (i,)

			blob1, adj1, img1, categ1, imgcurrent = build_image_graph(img_path)
			global_graph_live.append(empty)
			random_walk_desc_live.append(empty)
			random_walk_sems_live.append(empty)

			info_list = format_seg(blob1, categ1, adj1, i, counter )
			print(len(info_list))
			info_list = check_with_previous_and_fuse_live(i, counter, info_list, imgcurrent)
			print(len(info_list))
			add_to_global_graph_live(i, counter, info_list)
			counter += 1

	call_random_walk_live()
	see_desc_live(0,0,0)
	
#get bbox for that cnt, and get corresponding depth bbox
#copy just that part of depth that is in contour, to a blank uint8 image
#find non zero vals in deopth, find sum of non zero vals, find avg
#do this for both
#find diff, is it same as odom?

def build_image_graph_aux(imgpath):
	print(imgpath)
	img_seg = cv2.imread(imgpath)
	print(img_seg.shape)
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
	return blob_list[0], img_seg


def depth_stats(first, second):
	imgd_path1 = synthia_live_folder_depth + '%06d.png' % (first,)
	img_path1 = synthia_live_folder + '%06d.png' % (first,)
	blob1, img1 = build_image_graph_aux(img_path1)
	cnt = blob1[0]['cnt']
	blank = np.zeros((760,1280,3), np.uint8)
	cv2.drawContours(blank, [cnt], 0, (255,255,255), -1)
	plt.imshow(blank)
	plt.show()
	img_d1 = cv2.imread(imgd_path1,-1)

	#print(img_d1.shape)
	#print(img_d1.dtype)
	#blank_gray1 = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)

	x,y,w,h = cv2.boundingRect(cnt)	



	base = np.zeros((760,1280), np.uint16)
	depth_single_channel_copy = img_d1[:,:,1]
	blank_single_channel_copy = blank[:,:,1]
	print(depth_single_channel_copy.dtype)
	print(depth_single_channel_copy.shape)
	depth_roi_single_channel = depth_single_channel_copy[y:y+w,x:x+h]
	blank_roi_single_channel = blank_single_channel_copy[y:y+w,x:x+h]
	base_roi_single_channel = base[y:y+w,x:x+h]

	print(depth_roi_single_channel.dtype)
	print(base_roi_single_channel.dtype)
	print(blank_roi_single_channel.dtype)
	print(blank_roi_single_channel.shape)
	print(blank_roi_single_channel)
	print(blank_roi_single_channel.astype(bool))
	blank_new = blank_roi_single_channel.astype(bool)
	print(blank_new.shape)
	print(base_roi_single_channel.shape)
	print(depth_roi_single_channel.shape)
	np.copyto( base_roi_single_channel, depth_roi_single_channel, where = blank_new) 


	#base_roi_single_channel[blank_roi_single_channel>0]  = depth_roi_single_channel[blank_roi_single_channel>0] 
	print(base_roi_single_channel)
	#time.sleep(10)
#now blank is a mask
#bgr or rgb?
#compute stats
	#cv2.bitwise_and( , ,mask=blank_gray1 )
	print('.........................')

	imgd_path2 = synthia_live_folder_depth + '%06d.png' % (second,)
	img_path2 = synthia_live_folder + '%06d.png' % (second,)
	blob2, img2 = build_image_graph_aux(img_path2)
	cnt = blob2[0]['cnt']
	blank = np.zeros((760,1280,3), np.uint8)
	cv2.drawContours(blank, [cnt], 0, (255,255,255), -1)
	plt.imshow(blank)
	plt.show()
	img_d2 = cv2.imread(imgd_path2)
	#img_d2_3 = cv2.cvtColor(img_d2, cv2.COLOR_GRAY2BGR)




def check_with_previous_and_fuse_live(frame_num, position_num, info_list, imgcurrent):
	imgcurrent2 = np.copy(imgcurrent)
	print('PREVIOUS AND FUSE')
	print(frame_num)
	print(position_num)
	#centroids less than some dist
	#same categ
	#odom-----
	#area needs to be around the same(slightly larger)
	#bounding box/convex hull roughly the same-----
	#contour overlap?

	ref_list = global_graph_live[position_num-1]
	#print('len ref list')
	#print(len(ref_list))
	for sem_categ in range(len(rgb_mapping)):
		
		if(info_list[sem_categ] and ref_list[sem_categ]):
			#print('not denied')
			counter= 0 
			for blob in info_list[sem_categ]:
				counter_ref = 0
				for blob_ref in ref_list[sem_categ]:
					dist = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
					area_ratio = blob['area']/(1.*blob_ref['area']) ##is this float division?
					#print('dist of centroid' +str(dist))
					#print('area ratio '+str(area_ratio))
					if(dist < thresh and area_ratio < area_thresh and area_ratio > 0.78):
						print('condition yes')
						"""
						if(sem_categ==0):
							cv2.drawContours(imgcurrent2, [blob['cnt']], 0, (255,255,255), 6)
							cv2.drawContours(imgcurrent2, [blob_ref['cnt']], 0, (0,255,0), 6) ####
							plt.imshow(imgcurrent2)
							plt.show()
						"""
						#if(blob_ref['parentOf']):
							#print('FUSING TO ONE THAT IS ALREADY A PARENT')
							#print('dist of centroid' +str(dist))
							#print('area ratio '+str(area_ratio))
						#add condition to see if not already fused
						if(blob_ref['fused']):
							if(blob_ref['fuseDepth']>6):
								counter_ref+=1
								print('prev blob depth>6')
								continue
							#print('GLOBAL AND LOCAL')
						# assign global and local parents
							print('FUSING to fused parent')
							fuse_info = blob_ref['fuseInfo']
							global_graph_live[fuse_info[0]][fuse_info[1]][fuse_info[2]]['parentOf'].append((position_num, sem_categ, counter)) #####  change
							#can optionally add local child info to local parent also 
							#
							blob_ref['localParentOf'].append((position_num, sem_categ, counter))
							blob['fuseDepth'] = blob_ref['fuseDepth']+1
							global max_fuse_depth
							if(blob['fuseDepth']>max_fuse_depth):
								max_fuse_depth = blob['fuseDepth']
							blob['fused'] =  True
							blob['fuseInfo'] = fuse_info
							blob['localParent'] = [position_num-1, sem_categ, counter_ref]
						else:
							print('FUSING')
							blob_ref['localParentOf'].append((position_num, sem_categ, counter))
							blob_ref['parentOf'].append((position_num, sem_categ, counter))
							blob['fused'] =  True
							blob['fuseInfo'] = [position_num-1, sem_categ, counter_ref]
							blob['localParent'] = [position_num-1, sem_categ, counter_ref]
							blob['fuseDepth'] = 1
						#cv2.line(imgcurrent, blob['center'], blob_ref['center'], (255,255,255), 2)
						#length = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
						#print('early line length is: '+str(length))
						cv2.circle(imgcurrent,blob['center'], 8, (255,255,255), 2)
						break
					counter_ref += 1
				counter +=1
			#recovery tactic here
			counter_ref = 0



			for blob_ref in ref_list[sem_categ]:
				if(not blob_ref['parentOf']):
					counter = 0
					for blob in info_list[sem_categ]:
						if(blob['fused']):
							dist = np.linalg.norm(np.array(blob['center'])-np.array(blob_ref['center']))
							area_ratio = blob['area']/(1.*blob_ref['area'])
							#if(sem_categ==0 and blob_ref['center'][0]<550 and blob_ref['center'][1]<550):
								#imgcurrent3 = np.copy(imgcurrent2)
								#cv2.drawContours(imgcurrent3, [blob_ref['cnt']], 0, (255,255,255), 6) ####
								#cv2.drawContours(imgcurrent3, [blob['cnt']], 0, (0,0,255), 6) ####
								#plt.imshow(imgcurrent3)
								#plt.show()

							if(dist < thresh and area_ratio < area_thresh and area_ratio > 0.78):
								#print('CHANGING')

								#unset previous parent's info!
								assert(blob['fused'])
								parent_tuple = blob['localParent']
								parent_blob = global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]


								if(len(parent_blob['localParentOf'])>2):
									"""
									if(sem_categ==0):
										print(dist)
										print(area_ratio)
										cv2.drawContours(imgcurrent, [blob['cnt']], 0, (255,255,255), 6)
										cv2.drawContours(imgcurrent, [blob_ref['cnt']], 0, (0,255,0), 6) ####
										cv2.drawContours(imgcurrent, [parent_blob['cnt']], 0, (0,0,255), 6)
										plt.imshow(imgcurrent)
										plt.show()
									"""
									if(parent_tuple == blob['fuseInfo']):
										to_remove = global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['parentOf'].index((position_num, sem_categ, counter))
										to_remove2 = global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'].index((position_num, sem_categ, counter))

										del global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['parentOf'][to_remove]
										del global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'][to_remove2]

										blob_ref['parentOf'].append((position_num, sem_categ, counter))
										blob_ref['localParentOf'].append((position_num, sem_categ, counter))
										blob['fused'] =  True
										blob['fuseInfo'] = [position_num-1, sem_categ, counter_ref]
										blob['localParent'] = [position_num-1, sem_categ, counter_ref]

									else:
										global_par = blob['fuseInfo'] #global parent
										
										to_remove = global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'].index((position_num, sem_categ, counter))
										to_remove2 = global_graph_live[global_par[0]][sem_categ][global_par[2]]['parentOf'].index((position_num, sem_categ, counter))

										del global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['localParentOf'][to_remove]
										del global_graph_live[global_par[0]][sem_categ][global_par[2]]['parentOf'][to_remove2]

										if(blob_ref['fused']):
										# assign global and local parents
											fuse_info = blob_ref['fuseInfo']
											global_graph_live[fuse_info[0]][fuse_info[1]][fuse_info[2]]['parentOf'].append((position_num, sem_categ, counter))						
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


						counter +=1
				counter_ref += 1


				#draw lines between centres and disp image
				for blob in info_list[sem_categ]:
					if blob['fused']:
						parent_tuple = blob['fuseInfo']
						#print(parent_tuple)
						cv2.line(imgcurrent, blob['center'],global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['center'], (255,255,255), 2)
						length = np.linalg.norm(np.array(blob['center'])-np.array(global_graph_live[parent_tuple[0]][sem_categ][parent_tuple[2]]['center']))
						#if(sem_categ==0 and blob['center'][0]<550 and blob['center'][1]<550):
							#print('line length is: '+str(length))



		#else:
		#	print('if denied')
	#disp image
	#print('displaying fused info image')
	#plt.imshow(imgcurrent)
	#plt.show()
	return info_list					


def add_to_global_graph_live(frame_num, position_num, blob_list): #blobs list acc to semantic categs, with blob infos
	global_graph_live[position_num] = blob_list
	print(len(blob_list))
	random_walk_desc_live[position_num] = [[[] for blob in semcatblobs] for semcatblobs in blob_list]
	random_walk_sems_live[position_num] = [[[] for blob in semcatblobs] for semcatblobs in blob_list]

	#print(len(blob_list))
	print('add to global graph at '+str(position_num))
	print(len(global_graph_live))
	#print(len(global_graph_live[position_num]))
	#if(position_num==0):
		#print(global_graph_live[0])
		#time.sleep(30)
	print('...........')


def call_random_walk_live():
	counter_frame = 0
	for frame_list in global_graph_live:
		for sem_cat in range(len(rgb_mapping)):
			counter = 0
			for blob in frame_list[sem_cat]:
				if((not blob['fused']) and blob['edgeInfo']):
					print('calling for frame' + str(counter_frame))
					print('semcat '+str(sem_cat))
					print('calling for '+str(counter_frame)+'...'+str(sem_cat)+'...'+str(counter))
					print(blob['edgeInfo'])
					random_walk_live(walk_length, walk_length, [], counter_frame, sem_cat, counter, False) ##do we need to pass here?
					if(len(random_walk_sems_live[counter_frame][sem_cat][counter])==0):
						print('stopping for '+str(counter_frame)+'...'+str(sem_cat)+'...'+str(counter))
						#time.sleep(20)
				counter +=1
		counter_frame +=1

	"""
	blob = global_graph_live[0][0][0]
	cnt = blob['cnt']
	blank = np.zeros((760,1280,3), np.uint8)
	cv2.drawContours(blank, [cnt], 0, (255,255,255), 3)
	plt.imshow(blank)
	plt.show()
	"""

	#for t in random_walk_desc_live[0][0][0]:
		#print(t)
	print(len(random_walk_desc_live[0][0][0]))
	print(Counter(random_walk_sems_live[0][0][0]))


def see_desc_live(ind1,ind2,ind3):
	blank_image = np.zeros((1900,300,3), np.uint8)
	blank_image[:,:] = (255,255,255)      # (B, G, R)
	x=0
	for desc in random_walk_desc_live[ind1][ind2][ind3][:200]:
		y=0
		#change rows
		for node in desc:
			#keep shifting columns
			#print('node')
			#print(node[1])
			blank_image[x:x+8,y:y+26] = synthia_semantic_values[rgb_mapping[node[1]]]
			y += 26

		x +=8
	#blank_image = cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)
	plt.imshow(blank_image)
	plt.show()


def random_walk_live(walk_left, walk_length, list_not, position_num, sem_categ, blob_num, fused_case = False):

	if(len(list_not)>=4):
		time.sleep(10)
	if(fused_case):
		assert(walk_left!=0)
	if (walk_left==0):
		list_not.append((position_num,sem_categ,blob_num))
		#print(list_not)
		#time.sleep(1)
		head = list_not[0]
		
		print('----------------')
		print('last list')
		print(list_not)
		print('----------------')
		#print(len(list_not))
		#print('walk ended '+str(head[0]) + '..' + str(head[1]) + '..' + str(head[2]))
		
		random_walk_desc_live[head[0]][head[1]][head[2]].append([])
		random_walk_sems_live[head[0]][head[1]][head[2]].append([])

		for el in list_not:
			random_walk_desc_live[head[0]][head[1]][head[2]][-1].append((el[0],el[1],el[2]))
			random_walk_sems_live[head[0]][head[1]][head[2]][-1].append(el[1])
		random_walk_sems_live[head[0]][head[1]][head[2]][-1] = tuple(random_walk_sems_live[head[0]][head[1]][head[2]][-1])
		return

	no_neigh = True


	blob = global_graph_live[position_num][sem_categ][blob_num]
	edge_list = blob['edgeInfo']
	print(edge_list)
	for edge in edge_list:
		print('edge')
		(frame_num, position, sem, blob_pos) = edge
		if([position, sem, blob_pos] not in list_not):
			edge_blob = global_graph_live[position][sem][blob_pos]
			if(edge_blob['fused']):
				#print(edge_blob)
				print('edge blob is fused')
				#time.sleep(10)
				continue
			no_neigh = False
			if(fused_case):
				list_not_temp = list_not[:]
				
				print('--------------------------')
				print('node is a fused child case')
				print('walk left '+str(walk_left))
				print('checking out '+str(position)+','+str(sem)+','+str(blob_pos))
				print(list_not_temp)

				print('--------------------------')
				
				random_walk_live(walk_left-1, walk_length, list_not_temp, position, sem, blob_pos, False) ## minus one?


			else:	
				list_not2 = list_not[:]
				list_not2.append([position_num,sem_categ,blob_num])
				
				print('--------------------------')
				print('walk left '+str(walk_left))
				print('appending '+str(position_num)+','+str(sem_categ)+','+str(blob_num)+' to the list')
				print('checking out '+str(position)+','+str(sem)+','+str(blob_pos))
				print(list_not2)

				print('--------------------------')
				
				random_walk_live(walk_left-1, walk_length, list_not2, position, sem, blob_pos, False)

	if(not fused_case):
		fuse_list = blob['parentOf']
		for children_info in fuse_list:
			no_neigh = False
			#child_blob = global_graph[children_info[0]][children_info[1]][children_info[2]]
			list_not2 = list_not[:]
			list_not2.append([position_num,sem_categ,blob_num])
			
			print('--------------------------')
			print('going to fused child')

			print('walk left '+str(walk_left))
			print('appending '+str(position_num)+','+str(sem_categ)+','+str(blob_num)+' to the list')
			print('checking out child '+str(children_info[0])+','+str(children_info[1])+','+str(children_info[2]))
			print(list_not2)
			print('--------------------------')
			
			random_walk_live(walk_left, walk_length, list_not2, children_info[0],children_info[1],children_info[2], True)

	if(no_neigh):
		print('NO NEIGHBOURS')
		#time.sleep(10)
		return 

def compute():
	for i in range(len(random_walk_sems_memory)):
		for j in range(len(random_walk_sems_live)):
			for k in range(len(rgb_mapping)):
				for l in range(len(random_walk_sems_memory[i][k])):
					for g in range(len(random_walk_sems_live[j][k])):

						#print(global_graph[i][k][l])
						#print(global_graph[j][k][g])
						if((not global_graph[i][k][l]['fused']) and (not global_graph_live[j][k][g]['fused'])):
							print('.....................')
							print('semcat '+str(k))
							print('i j '+str(i)+' , '+str(j))
							print('l g '+str(l)+' , '+str(g))
							compute_matching(i,k,l,j,k,g)
							print('.....................')

#depth_stats(4,5)



load_odom_data(memory_odom_path,True)
build_graph_synthia(memory_odom_list)

load_odom_data(live_odom_path, False)
build_live_graph_synthia(live_odom_list, 0, 90)


compute()
print(matches)
f=sorted(matches, key = lambda x: x[0])
for fi in f:
	print(fi)

print(max_fuse_depth)


def convnet_compute(blob1, blob2, frame1, frame2, bool_mem1, bool_mem2):
	#only for nodes that have a random walk descriptor, and btw nodes of same sem cat
	#compute for fusings also? avg over them?
	#decide thresh for similarity from aj

	x1,y1,w1,h1 = blob1['roi']
	x2,y2,w2,h2 = blob2['roi']
	if(bool_mem1):
		img_path1 = synthia_memory_folder + '%06d.png' % (frame1,)
		rgb_path1 = synthia_memory_folder_rgb + '%06d.png' % (frame1,)

	else:
		img_path1 = synthia_live_folder + '%06d.png' % (frame1,)
		rgb_path1 = synthia_live_folder_rgb + '%06d.png' % (frame1,)

	if(bool_mem2):
		img_path2 = synthia_memory_folder + '%06d.png' % (frame2,)
		rgb_path2 = synthia_memory_folder_rgb + '%06d.png' % (frame2,)

	else:
		img_path2 = synthia_live_folder + '%06d.png' % (frame2,)
		rgb_path2 = synthia_live_folder_rgb + '%06d.png' % (frame2,)

		img_seg1 = cv2.imread(img_path1)
		real_img1 = cv2.imread(rgb_path1)
		img_seg2 = cv2.imread(img_path2)
		real_img2 = cv2.imread(rgb_path2)

	
	#check bgr rgb once
	
	height1, width1, channels1 = real_img1.shape
	height2, width2, channels2 = real_img2.shape

	roi1 = real_img1[max(0,y1-20):min(height1,y1+h1+20),max(0,x1-20):min(width1,x1+w1+20)]
	roi2 = real_img2[max(0,y2-20):min(height2,y2+h2+20),max(0,x2-20):min(width2,x2+w2+20)]

	#computing embeddings + grp +match
