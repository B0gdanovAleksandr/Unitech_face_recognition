# -*- coding: utf-8 -*-

import time
import contextlib
import cv2	
import numpy as np



@contextlib.contextmanager
def timex(verbose=False):
	"""
	Функция вычисления времени
	"""
	start = time.perf_counter()
	result = [start]
	try:
		yield result
	finally:
		result[0] = time.perf_counter() - start
		if verbose:
			print('\033[91mВремя обработки: {0:.3f} с\033[0m'.format(*result))


@contextlib.contextmanager
def time_debug(results):
	"""
	Функция вычисления времени
	"""
	start = time.perf_counter()
	try:
		yield results
	finally:
		results.append( time.perf_counter() - start )


class Viewer:
	
	def __init__(self, window_name='video capture', resize_factor=0):
		self.window_name = window_name
		self.resize_factor = resize_factor
		cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
		cv2.moveWindow(self.window_name, 0, 0)

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		cv2.destroyAllWindows()
	
	def draw_bboxes(self, rgb, metrics_list, id_color=None, crop_offset=np.array([0]*4)):
		if id_color is None:
			for face in metrics_list:
				self.draw_bbox(rgb, face['bbox'] + crop_offset, det_score=face.get('det_score'))
			return
		id, _color = id_color
		for i, face in enumerate(metrics_list):
			color = _color if i == id else (255,0,0)
			self.draw_bbox(rgb, face['bbox'] + crop_offset, color, det_score=face.get('det_score'))
	
	def draw_bbox(self, frame, bbox, color=(0,255,0), line_thickness=2, det_score=None):
		# Нарисовать прямоугольник поверх frame с координатами (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (B,G,R), line_thickness
		bbox = bbox.astype(int)
		cv2.rectangle(frame, bbox[:2], bbox[2:], color, line_thickness)
		if det_score:
			cv2.putText(frame, "%02.2f" % det_score, bbox[2:], cv2.FONT_HERSHEY_SIMPLEX, 1, color, line_thickness, cv2.LINE_AA)

	def imshow(self, rgb):
		frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		if self.resize_factor:
			frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)
		cv2.imshow(self.window_name, frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			return False
		if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0:
			return False
		return True

