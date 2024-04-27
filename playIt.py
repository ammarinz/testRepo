import random
from moveIt import PoseEstimation
import cv2 as cv
import mediapipe as mp
import sys
import os


def click_event(event, x, y, flags, param):
    """Method to get mouse position on click"""

    global mouse_x, mouse_y
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y


class playGame:

    def __init__(self):
        """Initialises attributes of PoseEstimation class"""

        # Dictionary for keys to body parts
        self. Ch_dict = {0: 'nose', 2: 'left Eye', 5: 'right eye', 11: 'left shoulder', 12: 'right shoulder',
                   13: 'left elbow', 14: 'right elbow', 15: 'left wrist', 16: 'right wrist', 25: 'left knee',
                   26: 'right knee', 29: 'left heel', 30: 'right heel'}
        self.challenge_list = list(self.Ch_dict.keys())   # list of dictionary keys

        self.max_number_of_rounds = 5                     # Number of playing rounds

        # On screen text settings
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.color = (0, 0, 0)
        self.thickness = 2
        self.font_scale = 1

    def display_final_message(self, original_frame):
        """Method to display final text on image

        Args:
            original_frame: Extracted frame from video
        """
        final_message = 'Game Over!!!'
        # final_message.append('You have played '+ str(self.max_number_of_rounds) + ' rounds')

        coordinates = [760, 500]                                        # Text locations are all chosen experimentally
        image = cv.putText(original_frame, final_message, coordinates, self.font, self.font_scale,
                           self.color, self.thickness, cv.LINE_AA)

        message = 'Final score are on the bottom-left corner of the screen'
        coordinates = [450, 550]
        image = cv.putText(original_frame, message, coordinates, self.font, self.font_scale,
                           self.color, self.thickness, cv.LINE_AA)

        message = 'See you soon :)!!!'
        coordinates = [730, 600]
        image = cv.putText(original_frame, message, coordinates, self.font, self.font_scale,
                           self.color, self.thickness, cv.LINE_AA)

        cv.imshow('playIt', image)
        cv.waitKey(0)

    def update_screen_info(self, original_frame, frame_landmarks, correct_answers, challenge_key_id, number_of_rounds):
        """Method to update and show text on image, including score and round number and which body part to find

        Args:
            original_frame: Extracted frame from video
            frame_landmarks: Landmarks extracted from frame using medipipe
            correct_answers: counter of number of correct answers
            challenge_key_id: Index to key body parts list which will be used to generate the questions
            number_of_rounds: Counter to the number of rounds/guesses

        Returns:
             image: numpy array for the image with the updates text information
        """

        top_left_text = 'Click on: '
        coordinates_top_left = [10, 30]
        image = cv.putText(original_frame, top_left_text, coordinates_top_left, self.font, self.font_scale,
                           self.color, self.thickness, cv.LINE_AA)

        if frame_landmarks is not None:
            color = (0, 0, 255)
            coordinates_top_left = [coordinates_top_left[0] + len(top_left_text)+130, 30]
            top_left_text = self.Ch_dict[self.challenge_list[challenge_key_id]]
            image = cv.putText(original_frame, top_left_text, coordinates_top_left, self.font,
                               self.font_scale, color, self.thickness, cv.LINE_AA)

        bottom_left_text = 'You have answered : ' + str(correct_answers) + ' out of ' + str(self.max_number_of_rounds) \
                           + ' correctly'

        coordinates_bottom_right1 = [10, 1000]
        cv.putText(original_frame, bottom_left_text, coordinates_bottom_right1, self.font, self.font_scale,
                   self.color, self.thickness, cv.LINE_AA)

        bottom_left_text = 'Round: ' + str(number_of_rounds)
        coordinates_bottom_right2 = [10, 1040]
        image = cv.putText(original_frame, bottom_left_text, coordinates_bottom_right2, self.font, self.font_scale,
                           self.color, self.thickness, cv.LINE_AA)
        return image

    def display_welcome_message(self, image_height, image_width, original_frame):

        text = ['Welcome to Guess the body part game, to play please follow these instructions:',
                '1- Wait until you see the required body part on the top-left corner of this screen.',
                '2- Click on the designated body part, one click is  enough.',
                '3- If your guess is correct, you will get 1 point.', '4- You will have 5 rounds.',
                '5- Your score will be displayed on the bottom-left corner of the window.',
                'Ready? Press any button to start.', 'Good luck!!! :)']

        for i in range(len(text)):
            coordinates = [int(image_width/6), 70 + 80*i]

            img = cv.putText(original_frame, text[i], coordinates, self.font, self.font_scale, self.color,
                             self.thickness, cv.LINE_AA)

        cv.imshow('playIt', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def update_challenge_coordinates(self, challenge_key_id, frame_landmarks, image_height, image_width):
        """Method to keep track of landmarks coordinates

           Args:
               challenge_key_id: Index to the key body part list
               frame_landmarks: Frame landmarks detection details
               image_height: Integer value representing frame height
               image_width: Integer value representing frame width

               Returns:
                   challenge_x: Updated landmark position in the x-direction
                   challenge_y: Updated landmark position in the y-direction

             """

        if frame_landmarks is None:
            return -1, -1

        challenge_x, challenge_y = frame_landmarks[self.challenge_list[challenge_key_id]].x, \
            frame_landmarks[self.challenge_list[challenge_key_id]].y
        challenge_x = challenge_x * image_width
        challenge_y = challenge_y * image_height

        return challenge_x, challenge_y

    def start_play(self, pose_estimator_game, path_to_video):
        """Method that includes most of the code for playIt game, it extracts frames from video, sends
           frames for landmark detection, compares click positions with required answer

           Args:
               pose_estimator_game: An object of PseEstimation class
               path_to_video: Full path to the video to be used in the game

           """
        global mouse_x, mouse_y
        correct_answers = 0                 # Counter of the number of correct answers
        number_of_rounds = 1                # Counter to the number of rounds/guesses so far

        range_of_key_landmark_random_numbers = len(self.challenge_list)-1

        # Extract frame 1 frame to obtain frame dimensions
        video_reader = cv.VideoCapture(path_to_video)
        _, original_frame = video_reader.read()
        image_height, image_width = original_frame.shape[0], original_frame.shape[1]
        video_reader.release()

        mp_drawing = mp.solutions.drawing_utils
        pose_estimator = mp.solutions.pose
        poses = pose_estimator.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        challenge_key_id = random.randint(0, range_of_key_landmark_random_numbers)
        challenge_key_id_previous = challenge_key_id       # Variable to store previous challenge id (which body part)

        video_reader = cv.VideoCapture(path_to_video)

        # Loop through video frames until the video is finished or the number of rounds reached maximum number
        while video_reader.isOpened() and number_of_rounds <= self.max_number_of_rounds:
            _, original_frame = video_reader.read()
            frame_id = int(video_reader.get(cv.CAP_PROP_POS_FRAMES))
            if frame_id == 1:
                self.display_welcome_message(image_height, image_width, original_frame)

            frame, frame_landmarks = pose_estimator_game.detect_landmarks(original_frame, pose_estimator, poses,
                                                                          mp_drawing)

            # Update and track landmark positions
            challenge_x, challenge_y = self.update_challenge_coordinates(challenge_key_id, frame_landmarks,
                                                                         image_height, image_width)

            image = self.update_screen_info(original_frame, frame_landmarks,
                                            correct_answers, challenge_key_id, number_of_rounds)

            image = cv.imshow('playIt', image)
            cv.setMouseCallback('playIt', click_event)           # Click event to get mouse position

            if mouse_x > -1 and mouse_y > -1:
                if challenge_x - 15 < mouse_x < challenge_x + 15 and challenge_y - 15 < mouse_y < challenge_y + 15:
                    correct_answers += 1
                    challenge_key_id = random.randint(0, range_of_key_landmark_random_numbers)
                else:
                    # Loop to ensure the next challenge is not the same as the current
                    while challenge_key_id_previous == challenge_key_id:
                        challenge_key_id = random.randint(0, range_of_key_landmark_random_numbers)
                number_of_rounds += 1
                mouse_x = -1
                mouse_y = -1

            cv.waitKey(20)
            challenge_key_id_previous = challenge_key_id

        self.display_final_message(original_frame)


if __name__ == "__main__":

    mouse_x = -1                                 # Mouse x and y click positions
    mouse_y = -1
    pose_estimator_game = PoseEstimation()       # Create object from PoseEstimation class

    pose_estimator_game.mode = 'game'            # Set 'game' mode for pose_estimator

    path_to_video = r'D:\job assignments\Moveai-MLCV-Assessment\project\video\cam01_walking_01.mp4'

    play = playGame()

    # play.start_play(pose_estimator_game, path_to_video)

    # Get path to current folder
    path = os.getcwd()

    # Get path to video folder
    path_to_video_folder = os.path.join(path, 'video')

    if not os.path.exists(path_to_video_folder):
        print('There was an error finding the path!, \n'
              'please make sure the \'video\' folder exists')
        sys.exit()

    #path_to_video = os.path.join(path_to_video_folder, str(sys.argv[1]))

    try:
        play.start_play(pose_estimator_game, path_to_video)
    except AttributeError:
        print('Folder does not have mp4 videos!,\nplease ensure the video is there.')
        sys.exit()



























