import numpy as np
import json
import cv2
import os
import math
import bbox_predictor
import random
import track

grid_size = 32
area_limited = [4000, 3600, 3800, 3600, 3600, 15000, 3800, 3200, 11000, 2000, 3600, 14000]
g_mask_idx = np.array(random.sample(range(0,64),24))
g_mask_idx = np.array([0,3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63])

def get_regions(img, boxes, labels, channel,dst_size=(64,64)):
	h = img.shape[0]
	w = img.shape[1]

	regions = []
	idx = 0

	for i, box in enumerate(boxes):
		if labels[i] == 1 or labels[i] == 2:
			continue
		y_min = int(math.floor(box[0] * h))
		x_min = int(math.floor(box[1] * w))
		y_max = int(math.floor(box[2] * h))
		x_max = int(math.floor(box[3] * w))

		b_h = y_max - y_min
		b_w = x_max - x_min
		if b_h * b_w < 3200:
			continue

		#if (b_h * b_w < area_limited[channel]):
		#	continue

		region = img[y_min:y_max, x_min:x_max, :]
		item = dict()
		item['data'] = cv2.resize(region, dst_size)
		item['crd'] = [y_min, x_min, y_max, x_max]
		item['center'] = [float((x_max + x_min))/2.0, float((y_max + y_min)) / 2.0]
		item['idx'] = idx
		item['mask_idx'] = g_mask_idx
		item['disappear'] = -1
		item['interval'] = 0

		regions.append(item)
		idx += 1
	return regions



if __name__ == "__main__":

	out_video_size = (960, 540)

	foler_idx = 0
	interval = 4
	video_path = "/DATACENTER2/ji.zhang/gs_test/danger/A47_20181015092844.mp4"
	det_dir = "/DATACENTER2/ji.zhang/gs_test/danger/A47_20181015092844_m_det"
	#output_dir = "/DATACENTER2/ji.zhang/gs_test"
	
	capture = cv2.VideoCapture(video_path)
	tot_frame = int(capture.get(7))
	frame_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	name_idx = 1
	predictor = bbox_predictor.Predictor()
	tracker = track.Tracker(frame_h, frame_w, grid_size)
	
	event_dir = "/DATACENTER2/ji.zhang/gs_test/tune"
	if not os.path.exists(event_dir):
		os.mkdir(event_dir)
	out_video = os.path.join(event_dir, 'vq_arct_lict.avi')
	frame_size=(720, 540)
	vw=cv2.VideoWriter(out_video,cv2.VideoWriter_fourcc("M","J","P","G"),25,frame_size,True)
	
	img = []
	print("start ----------------------------" + video_path)
	while True:
		rval, frame = capture.read()
		if rval == False:
			print(video_path + "   Done!!!!!!!!!!!!!!!!!")
			break
		
		if name_idx % interval != 1:
			img.append(frame)
			name_idx += 1
			continue

		last = os.path.join(det_dir, str(name_idx) + ".json")
		last_img = frame.astype(np.float32)
		last_json_str = open(last,'r').read()
		last_bbox = json.loads(last_json_str)['boxes']
		last_label = json.loads(last_json_str)['labels']
		last_regions = get_regions(last_img, last_bbox, last_label, foler_idx, dst_size=(64,64))

		predictor.region_match(last_regions)
		predict_bboxes = predictor.predict(last_regions, interval)

		tracker.track(last_regions)
		tracker.count_vehicles(frame_h, frame_w)

		road_state = "norml"
		#road_state, stop_vehicles, retro_vehicles = tracker.get_road_info()
				
		if name_idx > 1:
			for i, bboxes in enumerate(predict_bboxes):
				result_img = tracker.draw_count(img[i], True, road_state, bboxes)
				cv2.imwrite(os.path.join(event_dir, "count_" + str(name_idx - interval + i + 1) + ".jpg"), result_img)
				result_img = cv2.resize(result_img, frame_size)
				vw.write(result_img)

		result_img = tracker.draw_count(last_img, False, road_state)
		cv2.imwrite(os.path.join(event_dir, "count_" + str(name_idx) + ".jpg"), result_img)
		result_img = cv2.resize(result_img, frame_size).astype(np.uint8)
		vw.write(result_img)
				# results = {
				# "road_state" : road_state,
				# "up_road_count" : up_cnt,
				# "down_road_count" : down_cnt,
				# "stop_vehicle_uuids": stop_vehicles,
				# "vehicle_uuids" : uuids,
				# "vehicle_state" : states,
				# "vehicle_boxes" : boxes,
				# "vehicle_counts" : counts,
				# "vehicle_directions" : directions
				# }
				
				# with open(event_json, 'w') as f:
				# 	json.dump(results, f)
		predictor.update_src(last_img, last_regions)
		if name_idx % 100 == 0:
			print(name_idx, "/", tot_frame)
		name_idx += 1
		img = []
