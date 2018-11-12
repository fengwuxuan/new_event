import cv2

import os

import numpy as py

class Context():
	
	def __init__(self, img_h, img_w, grid_size):
		self.img_h = img_h
		self.img_w = img_w
		self.grid_size = grid_size
		self.h_step = int(self.img_h / grid_size) + 1
		self.w_step = int(self.img_w / grid_size) + 1
		self.grid_center = []
		self.grid_count = []
		self.grid_idx = []
		
		for h in range(0, self.h_step):
			for w in range(0, self.w_step):
				self.grid_center.append([int((w + 0.5)  * grid_size), int((h + 0.5) * grid_size)])
				self.grid_count.append(0)

	def update_context(self, vehicle_center, vehicle_direction):
		x = vehicle_center[0]
		y = vehicle_center[1]
		x = int(x / self.grid_size)
		y = int(y / self.grid_size)
		idx = x + y * self.w_step
		if vehicle_direction == 'up':
			self.grid_count[idx] += 1
		elif vehicle_direction == 'down':
			self.grid_count[idx] -= 1
		
		if self.grid_count[idx] > 4 and vehicle_direction == 'down':
			return True
		elif self.grid_count[idx] < -4 and vehicle_direction == 'up':
			return True
		
		return False