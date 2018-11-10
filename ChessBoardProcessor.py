"""Summary
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
from itertools import product
from sklearn.linear_model import LinearRegression


class ChessBoardProcessor(object):
    """docstring for ChessBoardProcessor

    ChessbOardProcessor is an object that takes a baseline empty board
    and generates a perspective transform from it. This can than be used
    to transform and split other boards taken from the same baseline camera 
    position

    Attributes:
        M (Numpy Array): Perspective Transform Array
    """

    def __init__(self, blank_board=None):
        """Summary
		
		Creates object and determines perspective transform

        Args:
            blank_board (None, optional): Path to blank board, if not passed,
            loads stored perspective transform from data folder
        """
        super(ChessBoardProcessor, self).__init__()

        if blank_board == None:
            # If no blank_board is specified, load perspective transform
            # from file
            self.M = np.load('data/M.npy')
            # print(self.M)
        else:
            self.M = self.perspective_transform(blank_board)
            # print(self.M)

    def perspective_transform(self, board_file):
        """Summary

		Calculates persepctive transform

        Args:
            board_file (PATH): Path to Blank Chess Board image 

        Returns:
            M (Numpy Array): Contains perspective transform array 
        """
        board_img = cv2.imread(board_file, 0)

        # Debug Image Show
        # plt.figure(figsize=(10,10))
        #plt.imshow(board_img, cmap='gray')
        # plt.show()

        # Find Inner Corners using OpenCV Chessboard Corners
        ret, corners = cv2.findChessboardCorners(
            board_img, (7, 7), flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH)

        if ret == False:
            # replace this with exception handling
            print('No Chessboard Found')
            print(len(corners))
            print(corners)
            return

        # Reshape Corners to make easier to work with
        corners = corners.reshape((49, 2))

        # Linear Regression to find four outer corners
        # Create X_train for linear regression
        a = np.linspace(7, 1, 7)
        b = np.linspace(7, 1, 7)
        X_train = np.array(list(product(a, b)))

        X_Regressor = LinearRegression()
        Y_Regressor = LinearRegression()
        X_Regressor.fit(X_train, corners[:, 0])
        Y_Regressor.fit(X_train, corners[:, 1])

        corner_features = np.array([
            (8, 8),
            (8, 0),
            (0, 0),
            (0, 8),
        ])

        x_predict = X_Regressor.predict(corner_features)
        y_predict = Y_Regressor.predict(corner_features)

        current_corners = np.float32([[x_predict[0], y_predict[0]], [x_predict[1], y_predict[1]], [
                                     x_predict[2], y_predict[2]], [x_predict[3], y_predict[3]]])
        transform_corners = np.float32(
            [[1080, 1080], [1080, 0], [0, 0], [0, 1080]])

        M = cv2.getPerspectiveTransform(current_corners, transform_corners)
        # save M to file
        np.save('data/M.npy', M)
        trans_board_img = cv2.warpPerspective(board_img, M, (1080, 1080))

        # Debug Image Show
        # plt.figure(figsize=(10,10))
        #plt.imshow(trans_board_img, cmap='gray')
        # plt.show()
        return M

    def board_split(self, board_file, state=0):
        """Summary

        Perform perspective transform on board and split into pieces

        Args:
            board_file (TYPE): Description
            state (int, optional): Description
            #

        Returns:
            Dictionary: containing piece images by piece
        """
        # board_location_dictionary = {
        # 	(0,0):'a8', (0,1):'b8', (0,2):'c8', (0,3):'d8',
        # 	(0,4):'e8', (0,5):'f8', (0,6):'g8', (0,7):'h8',
        # 	(1,0):'a7', (1,1):'b7', (1,2):'c7', (1,3):'d7',
        # 	(1,4):'e7', (1,5):'f7', (1,6):'g7', (1,7):'h7',
        # 	(2,0):'a6', (2,1):'b6', (2,2):'c6', (2,3):'d6',
        # 	(2,4):'e6', (2,5):'f6', (2,6):'g6', (2,7):'h6',
        # 	(3,0):'a5', (3,1):'b5', (3,2):'c5', (3,3):'d5',
        # 	(3,4):'e5', (3,5):'f5', (3,6):'g5', (3,7):'h5',
        # 	(4,0):'a4', (4,1):'b4', (4,2):'c4', (4,3):'d4',
        # 	(4,4):'e4', (4,5):'f4', (4,6):'g4', (4,7):'h4',
        # 	(5,0):'a3', (5,1):'b3', (5,2):'c3', (5,3):'d3',
        # 	(5,4):'e3', (5,5):'f3', (5,6):'g3', (5,7):'h3',
        # 	(6,0):'a2', (6,1):'b2', (6,2):'c2', (6,3):'d2',
        # 	(6,4):'e2', (6,5):'f2', (6,6):'g2', (6,7):'h2',
        # 	(7,0):'a1', (7,1):'b1', (7,2):'c1', (7,3):'d1',
        # 	(7,4):'e1', (7,5):'f1', (7,6):'g1', (7,7):'h1',
        # }

        board_location_dictionary = {
            (0, 0): 'h1', (0, 1): 'h2', (0, 2): 'h3', (0, 3): 'h4',
            (0, 4): 'h5', (0, 5): 'h6', (0, 6): 'h7', (0, 7): 'h8',
            (1, 0): 'g1', (1, 1): 'g2', (1, 2): 'g3', (1, 3): 'g4',
            (1, 4): 'g5', (1, 5): 'g6', (1, 6): 'g7', (1, 7): 'g8',
            (2, 0): 'f1', (2, 1): 'f2', (2, 2): 'f3', (2, 3): 'f4',
            (2, 4): 'f5', (2, 5): 'f6', (2, 6): 'f7', (2, 7): 'f8',
            (3, 0): 'e1', (3, 1): 'e2', (3, 2): 'e3', (3, 3): 'e4',
            (3, 4): 'e5', (3, 5): 'e6', (3, 6): 'e7', (3, 7): 'e8',
            (4, 0): 'd1', (4, 1): 'd2', (4, 2): 'd3', (4, 3): 'd4',
            (4, 4): 'd5', (4, 5): 'd6', (4, 6): 'd7', (4, 7): 'd8',
            (5, 0): 'c1', (5, 1): 'c2', (5, 2): 'c3', (5, 3): 'c4',
            (5, 4): 'c5', (5, 5): 'c6', (5, 6): 'c7', (5, 7): 'c8',
            (6, 0): 'b1', (6, 1): 'b2', (6, 2): 'b3', (6, 3): 'b4',
            (6, 4): 'b5', (6, 5): 'b6', (6, 6): 'b7', (6, 7): 'b8',
            (7, 0): 'a1', (7, 1): 'a2', (7, 2): 'a3', (7, 3): 'a4',
            (7, 4): 'a5', (7, 5): 'a6', (7, 6): 'a7', (7, 7): 'a8',
        }

        # Load Board and perform perpective transform
        board_img = cv2.imread(board_file, 0)
        trans_board_img = cv2.warpPerspective(board_img, self.M, (1080, 1080))
        directory = 'data/training_photos/unsorted/'
        count = 0

        (x1, x2, y1, y2) = (0, 135, 0, 135)

        if state == 1:
            board_img_dict = dict()

        for i in range(8):
            for j in range(8):
                sq_img = trans_board_img[x1:x2, y1:y2]
                if state == 1:
                    board_img_dict[board_location_dictionary[(i, j)]] = sq_img
                elif state == 0:
                    # self.label_image(sq_img)
                    name = directory + str(time.time()) + \
                        '_' + str(count) + '.jpg'
                    cv2.imwrite(name, sq_img)
                    count += 1
                    print(count)
                y1 += 135
                y2 += 135
            x1 += 135
            x2 += 135
            y1 = 0
            y2 = 135

        if state == 1:
            return board_img_dict

    def piece_display(self, piece_img):
        """Summary

		Helper function for plotting a piece

        Args:
            piece_img (array): 135x135 grayscale piece data 
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(piece_img, cmap='gray')
        plt.title('What type of piece is this?')
        plt.show()

    def label_image(self, piece_img):
        """Summary

        Prompt for label and save data

        Args:
            piece_img (array): 135x135 grayscale piece data
        """

        pieces_set = {'e', 'p', 'n', 'b', 'r', 'q', 'k'}
        colors_set = {'w', 'b'}

        self.piece_display(piece_img)
        # Accept User Input

        while(1):
            color = input("What color? (w - white, b - black) ")
            if color in colors_set:
                break
            else:
                print('Invalid Piece')
                self.piece_display(piece_img)

        while(1):
            piece = input(
                "What type of piece (e - empty, p - pawn, n - night, b - bishop, r - rook, q - queen, k - king: ")
            if piece in pieces_set:
                break
            else:
                print('Invalid Piece')
                self.piece_display(piece_img)

        if color == 'w':
            if piece == 'e':
                directory = 'data/training_photos/white_square/'
            elif piece == 'p':
                directory = 'data/training_photos/white_pawn/'
            elif piece == 'n':
                directory = 'data/training_photos/white_knight/'
            elif piece == 'b':
                directory = 'data/training_photos/white_bishop/'
            elif piece == 'r':
                directory = 'data/training_photos/white_rook/'
            elif piece == 'q':
                directory = 'data/training_photos/white_queen/'
            elif piece == 'k':
                directory = 'data/training_photos/white_king/'
        else:
            if piece == 'e':
                directory = 'data/training_photos/black_square/'
            elif piece == 'p':
                directory = 'data/training_photos/black_pawn/'
            elif piece == 'n':
                directory = 'data/training_photos/black_knight/'
            elif piece == 'b':
                directory = 'data/training_photos/black_bishop/'
            elif piece == 'r':
                directory = 'data/training_photos/black_rook/'
            elif piece == 'q':
                directory = 'data/training_photos/black_queen/'
            elif piece == 'k':
                directory = 'data/training_photos/black_king/'

        name = directory + str(time.time()) + '.jpg'
        cv2.imwrite(name, piece_img)
