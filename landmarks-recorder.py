import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2
import csv


MARGIN = 10  
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
NUM_HANDS = 2


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  for i in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[i]
    handedness = handedness_list[i]

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image


def write_to_csv(detection_result):
  print("HERE")
  hand_landmarks_list = detection_result.hand_landmarks

  for i in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[i]
    with open("./data/landmarks-dataset.csv", "a", newline="") as file:
      writer = csv.writer(file)
      writer.writerow([coord for landmark in hand_landmarks for coord in [landmark.x, landmark.y, landmark.z]])


def main():
  cam = cv2.VideoCapture(0)

  base_options = python.BaseOptions(model_asset_path="./models/hand_landmarker.task")
  options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=NUM_HANDS)
  detector = vision.HandLandmarker.create_from_options(options)

  while True:
    ret_val, image = cam.read()

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    annotated_image = image

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
      annotated_image = draw_landmarks_on_image(image, detection_result)
      # print(detection_result.hand_landmarks)
      # print(detection_result.handedness)
      # print()
    
    cv2.imshow("Webcam", cv2.flip(annotated_image, 1))
    cv2.resizeWindow("Webcam", 500, 450)

    key_pressed = cv2.waitKey(1)

    if key_pressed == 107: 
      if detection_result.hand_landmarks:
        write_to_csv(detection_result)

    if key_pressed == 27: 
      break  

  cam.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()