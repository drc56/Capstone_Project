"""Summary
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
from CNNModel import CNNModel

class GameBoardGenerator(object):

	"""Summary

	GameBoardGenerator object oversees piece predction and digital output generator
	
	Attributes:
	    board_img_dict (dictionary): Dictionary containing square images mapped to board position
	    board_piece_dict (dictionary): Dictionary containing predictions of piece at position
	    chess_model (CNNModel): Trained CNN Model for prediction
	"""
	
	def __init__(self, board_img_dict, chess_model):
		"""Summary
		
		Args:
		    board_img_dict (dictionary): Dictionary containing square images mapped to board position
		    chess_model (CNNModel): Trained CNN Model for prediction
		"""
		self.board_img_dict = board_img_dict
		self.board_piece_dict = dict()
		self.chess_model = chess_model



	def predict_pieces(self):
		"""Summary

		Predict piece and store results to board dict

		"""
		for key, value in self.board_img_dict.items():
			piece,color = self.chess_model.predict_image(value)
			self.board_piece_dict[key] = color+piece

	def display_board(self):
		"""Summary

		Print predicted board to terminal

		"""
		grid_order = ['a8','b8','c8','d8','e8','f8','g8','h8',
			'a7','b7','c7','d7','e7','f7','g7','h7',
			'a6','b6','c6','d6','e6','f6','g6','h6',
			'a5','b5','c5','d5','e5','f5','g5','h5',
			'a4','b4','c4','d4','e4','f4','g4','h4',
			'a3','b3','c3','d3','e3','f3','g3','h3',
			'a2','b2','c2','d2','e2','f2','g2','h2',
			'a1','b1','c1','d1','e1','f1','g1','h1']

		print('Predicted chess board: ')

		for pos in grid_order:
			if pos.find('a') == 0:
				print('|',end="")
			print(self.board_piece_dict[pos],end="")
			if pos.find('h') == 0:
				print('|')
			else:
				print('|',end="")




