import cv2 as cv
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import json
from sys import exit


class PoseEstimation:

    def __init__(self):
        """Initialises attributes of PoseEstimation class"""

        self.path = os.getcwd()
        self.mode = 'detection'              # Two modes available, 'detection', 'game'
        self.image_width = 1920              # Default image width
        self.image_height = 1080                  # Default image height

    @staticmethod
    def export_to_json(frame_landmarks, path_to_video, frame_id):
        """Export landmarks to json file, store files in 'json' folder, creates that folder if it does not exist

        Args:
            frame_landmarks: Landmarks detections of frame, None if no detections
            path_to_video: Path to the specific mp4 video, used to extract file name
            frame_id: Index to frame of the video
            """

        filename = os.path.splitext(os.path.basename(path_to_video))[0] + '.json'
        filename_full_path = os.path.join(os.getcwd(), 'json', filename)

        # Create json folder if it does not exist
        os.mkdir('json') if 'json' not in os.listdir() else None

        # This will overwrite the json file every new run
        open(filename_full_path, 'w') if frame_id == 1 else None

        if frame_landmarks is None:
            no_landmarks_dict = {'frameId': frame_id,
                                 'hasData': 'False'}
            with open(filename_full_path, 'a') as outfile:
                json.dump(no_landmarks_dict, outfile)
        else:
            for _, landmarks in enumerate(frame_landmarks):
                position = {'x': landmarks.x, 'y': landmarks.y, 'z': landmarks.z}
                landmarks_dict = {'frameId': frame_id,
                                  'hasData': 'True',
                                  'positions': position,
                                  'vis': landmarks.visibility}
                with open(filename_full_path, 'a') as outfile:
                    # json_object = json.dumps(landmarks_dict, indent=4)
                    json.dump(landmarks_dict, outfile)

    @staticmethod
    def detect_landmarks(frame, pose_estimator, poses, mp_drawing):
        """Detects landmarks per frame

        Args:
            frame: numpy array, BGR frame extracted from video
            pose_estimator: mediapipe module for pose
            poses: Object for pose estimation
            mp_drawing: module for drawing results

            Raises:
                AttributeError: If no landmark was detected

            Returns:
                 frame: frame with detected landmarks
                 frame_landmarks: Detection landmarks details( coordinates, visibility), return None if
                 no landmarks were detected
        """

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert frame to RGB
        results = poses.process(rgb_frame)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, pose_estimator.POSE_CONNECTIONS)

        try:
            results.pose_landmarks.landmark
        except AttributeError:
            return frame, None
        frame_landmarks = np.array(results.pose_landmarks.landmark)
        return frame, frame_landmarks

    def read_video(self, path_to_video, number_of_videos):
        """Reads video frame by frame and send each frame to another method for landmark detection

        Args:
            path_to_video: path to the mp4 video to be processed
            number_of_videos: number of videos that exist in the 'video' folder

        Returns:
            frame: frames with detection landmarks (only in 'game' mode)
            frame_landmarks: Detection landmarks details( coordinates, visibility) (only in 'game' mode)

            """

        # Create object for reading videos
        video_reader = cv.VideoCapture(path_to_video)

        # Creation and initialisation objects for pose detection
        mp_drawing = mp.solutions.drawing_utils
        pose_estimator = mp.solutions.pose
        poses = pose_estimator.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Loop to extract video frames and send them for landmark detection
        while video_reader.isOpened():
            _, frame = video_reader.read()
            frame_id = int(video_reader.get(cv.CAP_PROP_POS_FRAMES))

            # Get the dimension of video frames
            if frame_id == 1:
                self.image_width, self.image_height = frame.shape[1], frame.shape[0]

            # Downsize video frames by 4, if there are more than 1 video
            if number_of_videos > 1:
                frame = cv.resize(frame, (int(self.image_width/4), int(self.image_height/4)))

            # Send frames for landmark detection, then show it on screen
            frame, frame_landmarks = self.detect_landmarks(frame, pose_estimator, poses, mp_drawing)
            cv.imshow('moveIt', frame)
            cv.waitKey(1)
            self.export_to_json(frame_landmarks, path_to_video, frame_id)              # Export to json

            # Return data to playIt if 'game' mode is on
            if self.mode == 'game':
                return frame, frame_landmarks

    @staticmethod
    def create_video_list(path_to_video_folder):
        """Create video list from videos inside 'video' folder
        Args:
            path_to_video_folder: path to 'video' folder

        Raises:
            IOError: if 'video' folder does not exist.

        Returns:
            A list of mp4 videos available in 'video' folder, returns
            an empty list if mo videos available
            """

        video_list = []
        try:
            for file in os.listdir(path_to_video_folder):
                if file.endswith('.mp4'):
                    video_list.append(file)
        except IOError:
            print('There was an error finding the path!, \n'
                  'please make sure the \'video\' folder exists')
            exit()

        return video_list

    def prepare_for_run(self):
        """Prepare paths of videos"""

        # Prepare videos path
        path_to_video_folder = os.path.join(self.path, 'video')         # Get path to video folder

        # Find number of videos and create video list
        video_file_list = self.create_video_list(path_to_video_folder)
        number_of_videos = len(video_file_list)
        if number_of_videos == 0:
            print('Folder does not have mp4 videos!,\nplease ensure videos are there.')
            exit()

        with ProcessPoolExecutor(max_workers=number_of_videos) as pool:
            for _, file in enumerate(video_file_list):
                path_to_video = os.path.join(path_to_video_folder, file)
                pool.submit(self.read_video, path_to_video, number_of_videos)


if __name__ == "__main__":

    # Create object and start moveIt app
    pose = PoseEstimation()
    pose.prepare_for_run()
