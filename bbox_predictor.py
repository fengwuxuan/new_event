import numpy as np

import os

import json

import cv2

import math

import time

import random



def extract_feature_hand(region):

	hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)



	color = ('h', 's', 'v')

	hists = []

	for idx, col in enumerate(color):

		hist = cv2.calcHist([hsv], [idx], None, [64], [0, 1])

		hists.append(hist)

	ff = np.hstack((hists[0], hists[1], hists[2])).reshape(192)



	tmp = (cv2.resize(region, (32,32)) - 128.0)/255.0

	tmp = tmp.reshape(32*32*3)

	return np.hstack((tmp, ff)).reshape(32*32*3 + 192)



def extract_feature_mask(region, mask_idx):

	tmp = cv2.resize(region, (64, 64))

	tmp = (tmp.reshape((64,8,8,3)) - 128) / 255

	return tmp[mask_idx].reshape(len(mask_idx) * 8 * 8 * 3)



def find_neigbor_regions(probe, candidants, search_rad=0):

	re = []

	search_rad_ = 0

	w = float(probe['crd'][3] - probe['crd'][1])

	h = float(probe['crd'][2] - probe['crd'][0])

	#print(probe['idx'], ":", w, "===========", h)
	#print(probe['crd'][0], probe['crd'][1], probe['crd'][2], probe['crd'][3])
	ratio = w / h

	if search_rad == 0:

		search_rad_ = math.sqrt(w**2 + h**2) * 0.60 * ratio

		probe['search_rad'] = search_rad_

	for idx, candidant in enumerate(candidants):

		dist = cal_dist(np.array(probe['center']), np.array(candidant['center']))

		if dist < search_rad_:

			re.append(candidant)

	return re



def cal_dist(a, b, dist_type = 'eu'):

	if dist_type == 'eu':

		return np.linalg.norm(a - b)

	elif dist_type == 'cos':

		return np.dot(a,b)/(np.linalg.norm(a)*(np.linalg.norm(b)))



def match(probe, candidants):

	max_dist = -1.0

	max_idx = -1

	p_w = float(probe['crd'][3] - probe['crd'][1])

	p_h = float(probe['crd'][2] - probe['crd'][0])

	p_cnt = probe['center']

	p_area = p_w * p_h



	for idx, candidant in enumerate(candidants):

		c_w = float(candidant['crd'][3] - candidant['crd'][1])

		c_h = float(candidant['crd'][2] - candidant['crd'][0])

		c_cnt = candidant['center']

		c_area = c_w * c_h



		ratio_v = 1 - (math.fabs(p_area - c_area)/ math.fabs(p_area + c_area))



		delta_x = c_cnt[0] - p_cnt[0]

		delta_y = c_cnt[1] - p_cnt[1]



		dist = cal_dist(probe['feature'], candidant['feature'], 'cos') * (ratio_v * 0.75 + 1)

		mask_dist = cal_dist(probe['mask_feature'], candidant['mask_feature'], 'cos')



		dist = dist + mask_dist * 0.8



		if dist > max_dist:

			max_dist = dist

			max_idx = idx

	return max_idx, max_dist





class Predictor():

	def __init__(self):
		# sself.img
		self.src_regions = []
		self.count = 0

	
	def region_match(self, dst_regions):
		for item in dst_regions:
			item['feature'] = extract_feature_hand(item['data'])
			item['mask_feature'] = extract_feature_mask(item['data'], item['mask_idx'])
		
		#match
		for idx, probe in enumerate(self.src_regions):
			neigbor_candidants = find_neigbor_regions(probe, dst_regions)
			if len(neigbor_candidants) > 0:
				match_idx, match_dist = match(probe, neigbor_candidants)
			
				if match_idx == -1:
					continue
			
				if 'match' not in neigbor_candidants[match_idx]:
					probe['match'] = neigbor_candidants[match_idx]['idx']
					probe['match_scores'] = match_dist

					neigbor_candidants[match_idx]['match'] = probe['idx']
					neigbor_candidants[match_idx]['match_scores'] = match_dist
				else:
					if match_dist > neigbor_candidants[match_idx]['match_scores']:
						probe['match'] = neigbor_candidants[match_idx]['idx']
						probe['match_scores'] = match_dist

						self.src_regions[neigbor_candidants[match_idx]['match']].pop('match', None)
						self.src_regions[neigbor_candidants[match_idx]['match']].pop('match_scores', None)

						neigbor_candidants[match_idx]['match'] = probe['idx']
						neigbor_candidants[match_idx]['match_scores'] = match_dist

		for region in dst_regions:
			if 'match' not in region:
				region['uuid'] = self.count
				self.count = self.count + 1
			else:
				region['uuid'] = self.src_regions[region['match']]['uuid']
	
	def predict(self, dst_regions, interval):
		bboxes_list = []
		for i in range(1, interval):
			regions_crd = []
			for idx, probe in enumerate(self.src_regions):
				if 'match' not in probe.keys() or probe['match'] == None:
					continue
				match_id = probe['match']
				deltaY_1 =  float(dst_regions[match_id]['crd'][0] - probe['crd'][0]) / interval
				deltaX_1 =  float(dst_regions[match_id]['crd'][1] - probe['crd'][1]) / interval
				deltaY_2 =  float(dst_regions[match_id]['crd'][2] - probe['crd'][2]) / interval
				deltaX_2 =  float(dst_regions[match_id]['crd'][3] - probe['crd'][3]) / interval

				predict_Y_1 = int(probe['crd'][0] + deltaY_1 * i)
				predict_X_1 = int(probe['crd'][1] + deltaX_1 * i)
				predict_Y_2 = int(probe['crd'][2] + deltaY_2 * i)
				predict_X_2 = int(probe['crd'][3] + deltaX_2 * i)

				regions_crd.append([predict_Y_1, predict_X_1, predict_Y_2, predict_X_2, probe["uuid"]])
			bboxes_list.append(regions_crd)
		return bboxes_list


	def update_src(self, dst_img, dst_regions):

		self.img = dst_img

		self.src_regions = dst_regions

		for region in self.src_regions:

			region.pop('match', None)

			region.pop('match_scores', None)



	def draw_match(self, dst_img, dst_regions):

		h = self.img.shape[0]

		cavans = np.vstack((self.img, dst_img))

		red = (0, 0, 255) #8

		blue = (255, 0, 0)

		font = cv2.FONT_HERSHEY_SIMPLEX

	

		for idx, regions in enumerate(self.src_regions):

			y_min, x_min, y_max, x_max = regions['crd'][0], regions['crd'][1], regions['crd'][2], regions['crd'][3]

			uuid = regions["uuid"]

			cv2.rectangle(cavans, (x_min, y_min), (x_max, y_max), blue, 2)

			cv2.putText(cavans, str(uuid), (x_min, y_min), font, 2, (0,0,255), 2, cv2.LINE_AA)



		for idx, regions in enumerate(dst_regions):

			y_min, x_min, y_max, x_max = regions['crd'][0], regions['crd'][1], regions['crd'][2], regions['crd'][3]

			uuid = regions["uuid"]

			cv2.rectangle(cavans, (x_min, y_min + h), (x_max, y_max + h), blue, 2)

			cv2.putText(cavans, str(uuid), (x_min, y_min + h), font, 2, (0,0,255), 2, cv2.LINE_AA)



		for idx, probe in enumerate(self.src_regions):

			src_crd = (int(probe['center'][0]), int(probe['center'][1]))

			#print(probe)

			if 'match' not in probe.keys() or probe['match'] == None:

				continue

			match_region = dst_regions[probe['match']]

			dst_crd = (int(match_region['center'][0]), int(match_region['center'][1] + h))

			cv2.line(cavans, src_crd, dst_crd, red)

		return cavans



	def draw_bbox(self, prefix, bboxes_list, name_idx):

		blue = (255, 0, 0)

		font = cv2.FONT_HERSHEY_SIMPLEX



		images = []

		for i, bboxes in enumerate(bboxes_list):

			img =cv2.imread(os.path.join(prefix, str(name_idx + i + 1) + '.jpg')).astype(np.float32) 



			for idx, bbox in enumerate(bboxes):

				y_min, x_min, y_max, x_max, uuid = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

				cv2.rectangle(img, (x_min, y_min), (x_max, y_max), blue, 2)

				cv2.putText(img, str(uuid), (x_min, y_min), font, 2, (0,0,255), 2, cv2.LINE_AA)

			images.append(img)

		return images