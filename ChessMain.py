from ChessBoardProcessor import ChessBoardProcessor
from CNNModel import CNNModel
import cv2
import os

def main():

	chess_model = CNNModel(model_version=0)
	chess_model.compile_model()
	chess_model.train_model(augment=0)
	chess_model.load_model_best_weights()
	chess_model.test_models()



    # set2_dir = 'data/board_photos/set2/'
    # set2_array = os.listdir(set2_dir)

    # baseline_board_path = set2_dir + 'baseline_board.jpg'
    # game_board = ChessBoardProcessor(blank_board=baseline_board_path)

    # # for file in set2_array:
    # # 	if file == 'baseline_board.jpg':
    # # 		continue
    # # 	else:
    # # 		board_img_path = set2_dir + file
    # # 		game_board.board_split(board_img_path, state=0)

    # unsorted_dir = 'data/training_photos/unsorted/'
    # unsorted_array = os.listdir(unsorted_dir)
    # num_pics = len(unsorted_array)
    # value = 1

    # for file_path in unsorted_array:
    # 	full_file_path = unsorted_dir + file_path
    # 	piece_pic = cv2.imread(full_file_path,0)
    # 	game_board.label_image(piece_pic)
    # 	print('Image {} of {}'.format(value,num_pics))
    # 	value += 1

    # test_board_path = 'IMG_20181007_134352.jpg'
    # test_board_dict = game_board.board_split(test_board_path,1)

    # #check a few pieces
    # game_board.piece_display(test_board_dict['e4'])
    # game_board.piece_display(test_board_dict['e5'])
    # game_board.piece_display(test_board_dict['d4'])
    # game_board.piece_display(test_board_dict['d5'])
    # game_board.piece_display(test_board_dict['a1'])
    # game_board.piece_display(test_board_dict['a8'])
    # game_board.piece_display(test_board_dict['h1'])
    # game_board.piece_display(test_board_dict['h8'])

if __name__ == "__main__":
    main()
