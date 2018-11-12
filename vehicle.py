import numpy as py



import os

import math
import cv2
import context


class Vehicle(object):
	def __init__(self, region):
		self.uuid = region['uuid']
		self.crdque = []
		self.crdque.append(region['crd'])
		self.center = []
		self.center.append(region['center'])
		self.disappear = -1
		self.count = -1
		self.deltacnt = []
		self.mv_distance = []
		self.state = "normal"
		self.direction = ""
		self.retromotion = 0

	def get_uuid(self):
		return self.uuid


	def get_center(self):
		return self.center[-1]

	def set_disappear(self):
		self.disappear += 1

	def get_size(self):
		crd = self.crdque[-1]
		h = crd[2] - crd[0]
		w = crd[3] - crd[1]
		return [h, w]

	def get_direction(self):
		return self.direction

	def get_disappear(self):
		return self.disappear

	def get_count(self):
		return self.count

	def set_count(self, value):
		self.count = value

	def get_crd(self):
		return self.crdque

	def get_box(self):
		return self.crdque[-1]

	def get_state(self):
		return self.state

	def update_state(self):
		if len(self.center) < 5:
			self.state = "normal"
			return

		center_now = self.center[-1]
		center_pre = self.center[-4]

		mv_x = center_now[0] - center_pre[0]
		mv_y = center_now[1] - center_pre[1]

		dis = math.sqrt(mv_x * mv_x + mv_y * mv_y) / 4.0
		if dis < 1.5:
			self.state = "stop"
		elif dis < 3.0:
			self.state = "slowdown"
		else:
			self.state = "normal"
		return

	def get_retromove(self):
		return self.retromotion

	def update_info(self, region, context):
		self.isupdate = True
		self.crdque.append(region['crd'])
		self.disappear = -1

		mv_x = region['center'][0] - self.center[0][0]
		mv_y = region['center'][1] - self.center[0][1]

		if mv_y < -4.0:
			self.direction = "up"
		elif mv_y > 4.0:
			self.direction = "down"

		self.center.append(region['center'])
		self.update_state()
		
		if self.direction != "":
			isretro = context.update_context(region['center'], self.direction)
			if isretro == True:
				self.retromotion += 1
			else:
				self.retromotion = 0 