import cv2

import os

import numpy as py

import vehicle

import context

class Tracker():

	def __init__(self, img_h, img_w, grid_size):
		self.vehicles = []
		self.context = context.Context(img_h, img_w, grid_size)
		self.up_vehicle_count = 0
		self.down_vehicle_count = 0

	def get_queue(self):
		return self.vehicles

	def track(self, dst_regions):
		for region in dst_regions:
			flag = 0
			for idx in range(len(self.vehicles)):
				if region['uuid'] == self.vehicles[idx].get_uuid():
					self.vehicles[idx].update_info(region, self.context)
					flag = 1
					break
			if flag == 0:
				new_vehicle = vehicle.Vehicle(region)
				self.vehicles.append(new_vehicle)

		n = len(self.vehicles)
		for idx in xrange(n):
			if idx >= len(self.vehicles):
				break
			self.vehicles[idx].set_disappear()
			if self.vehicles[idx].get_disappear() >= 4:
				self.vehicles.pop(idx)
				idx -= 1

	def count_vehicles(self, img_h, img_w):
		for idx in range(len(self.vehicles)):
			# h, w = self.vehicles[idx].get_size()
			# # if h * w < 40 * 40:
			# # 	continue

			y = self.vehicles[idx].get_center()[1]
			if y < img_h / 2:
				continue
			
			if self.vehicles[idx].get_count() == -1:
				direction = self.vehicles[idx].get_direction()
				if direction == "up":
					self.up_vehicle_count += 1
					self.vehicles[idx].set_count(self.up_vehicle_count)
				elif direction == "down":
					self.down_vehicle_count += 1
					self.vehicles[idx].set_count(self.down_vehicle_count)
					

	def get_road_info(self):
		road_state = "normal"
		stop_vehicles = []
		retromove_vehicles = []
		num_stop = 0
		num_slowdown = 0
		
		for idx in range(len(self.vehicles)):
			vehicle_state = self.vehicles[idx].get_state()

			if vehicle_state == "stop":
				num_stop += 1
				num_slowdown += 1
				stop_vehicles.append(self.vehicles[idx].get_uuid())
			elif vehicle_state == "slowdown":
				num_slowdown += 1

			retromove = self.vehicles[idx].get_retromove()
			if retromove > 4:
				retromove_vehicles.append(self.vehicles[idx].get_uuid())

		if len(self.vehicles) > 8 and num_stop > 3:
			road_state = "congestion"

		elif len(self.vehicles) > 6 and num_slowdown > 4:
			road_state = "slowdown"

		return road_state, stop_vehicles, retromove_vehicles


	def get_vehiclesinfo(self, h, w):
		boxes = []
		states = []
		uuids = []
		counts = []
		directions = []
		for idx in range(len(self.vehicles)):
			if self.vehicles[idx].get_disappear() > 0:
				continue
			#print(idx)
			states.append(self.vehicles[idx].get_state())
			uuids.append(self.vehicles[idx].get_uuid())
			counts.append(self.vehicles[idx].get_count())
			directions.append(self.vehicles[idx].get_direction())
			box = self.vehicles[idx].get_box()
			y_min = float(box[0]) / h
			x_min = float(box[1]) / w
			y_max = float(box[2]) / h
			x_max = float(box[3]) / w
			boxes.append([y_min, x_min, y_max, x_max])
		return uuids, states, boxes, counts, directions, self.up_vehicle_count, self.down_vehicle_count

	
	def draw_count(self, img, is_interval, state, bboxes = []):

		blue = (255, 0, 0)

		green = (0, 255, 0)

		red = (0, 0, 255)

		unkonw = (0, 255, 255)

		retro = (255, 255, 0)

		font = cv2.FONT_HERSHEY_SIMPLEX

		color = blue

		h = img.shape[0]
		w = img.shape[1]
		cv2.line(img, (0, h / 2), (w, h / 2), color, 4)
		cv2.line(img, (0, h * 2 / 3), (w, h * 2 / 3), green, 4)


		cv2.putText(img, str(state), (4, 30), font, 2, (0,0,255), 2, cv2.LINE_AA)

		if is_interval:

			for idx, bbox in enumerate(bboxes):

				color = blue

				y_min, x_min, y_max, x_max, uuid = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

				for idx in range(len(self.vehicles)):

					if uuid == self.vehicles[idx].get_uuid():

						uuid = self.vehicles[idx].get_count()

						if self.vehicles[idx].get_direction() == "down":
							color = green

						if self.vehicles[idx].get_state() == "stop":
							color = red

						if self.vehicles[idx].get_retromove() > 2:
						 	color = retro
						break
				if uuid == -1 and cmp(color, red) != 0 and cmp(color, retro) != 0:
					color = unkonw

				cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
				cv2.putText(img, str(uuid), (x_min, y_min), font, 1, color, 1, cv2.LINE_AA)
		else:
			for idx in range(len(self.vehicles)):
				color = blue

				if self.vehicles[idx].get_disappear() > 0:
					continue
				y_min, x_min, y_max, x_max = self.vehicles[idx].get_crd()[-1]
				uuid = self.vehicles[idx].get_count()
				if self.vehicles[idx].get_direction() == "down":
					color = green
				if self.vehicles[idx].get_state() == "stop":
					color = red
				if self.vehicles[idx].get_retromove() > 2:
						 	color = retro
				if uuid == -1 and cmp(color, red) != 0 and cmp(color, retro) != 0:
					color = unkonw

				cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
				cv2.putText(img, str(uuid), (x_min, y_min), font, 1, color, 1, cv2.LINE_AA)
		return img