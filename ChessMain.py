from ChessBoardProcessor import ChessBoardProcessor
from CNNModel import CNNModel
from ChessBoardGenerator import GameBoardGenerator
import cv2
import os


def main():

	"""
	Flow for creating a game_board object with the baseline board in the test set
	"""
	test_dir = 'data/board_photos/test/'
	baseline_board_path = test_dir + 'baseline_board.jpg'
	game_board = ChessBoardProcessor(blank_board=baseline_board_path)

	"""
	Flow for creating a CNN object
	Training and tested are commented out but this is how it would be called
	Even after training the weights must be loaded into the model to get the best weights
	"""

	chess_model = CNNModel(model_version=1)
	chess_model.compile_model()
	#chess_model.train_model(augment=0)
	chess_model.load_model_best_weights()
	#chess_model.test_models()

	"""
	Two examples of loading a test image that has the same baseline, predicting the pieces
	and outputting to screen a board example
	"""
	test_board_path = 'data/board_photos/test/IMG_20181005_162420.jpg'
	test_board = GameBoardGenerator(game_board.board_split(test_board_path,1),chess_model)
	test_board.predict_pieces()
	test_board.display_board()

	test_board_path_2 = 'data/board_photos/test/IMG_20181005_162943.jpg'
	test_board2 = GameBoardGenerator(game_board.board_split(test_board_path_2,1),chess_model)
	test_board2.predict_pieces()
	test_board2.display_board()

if __name__ == "__main__":
	main()
