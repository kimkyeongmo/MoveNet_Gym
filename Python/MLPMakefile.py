# # # openai LLM import
# # # from openai import OpenAI
# # #
# # #
# #
# # #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# # # MoveNet Quantization + Model save
# import cv2
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
#
# # MoveNet 모델 로드
# model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
# movenet = model.signatures['serving_default']
# converter = tf.lite.TFLiteConverter.from_concrete_functions([movenet])
# #tf.float16 Quantization
#
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float32]
# tflite_fp16_model = converter.convert()
#
# with open("movenet_lightning_fp32.tflite", "wb") as f:
#     f.write(tflite_fp16_model)
#
# # #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# #
# # import pandas as pd
# # import kagglehub
# #
# #
# # # Download latest version
# # # path = kagglehub.dataset_download("mrigaankjaswal/exercise-detection-dataset")
# # #
# # # print("Path to dataset files:", path)
# #
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv(
    "mrigaankjaswal/exercise-detection-dataset/versions/1/exercise_angles.csv"
)

def angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def extract_angles(keypoints, side="right"):
    if side == "right":
        shoulder = keypoints[6][:2]
        elbow = keypoints[8][:2]
        wrist = keypoints[10][:2]
        hip = keypoints[12][:2]
        knee = keypoints[14][:2]
        ankle = keypoints[16][:2]
    else:
        shoulder = keypoints[5][:2]
        elbow = keypoints[7][:2]
        wrist = keypoints[9][:2]
        hip = keypoints[11][:2]
        knee = keypoints[13][:2]
        ankle = keypoints[15][:2]

    return {
        "Shoulder_Angle": angle(hip, shoulder, elbow),
        "Elbow_Angle": angle(shoulder, elbow, wrist),
        "Hip_Angle": angle(shoulder, hip, knee),
        "Knee_Angle": angle(hip, knee, ankle)
    }


dataset["Side"] = dataset["Side"].map({"left": 0, "right": 1})
ground_cols = [col for col in dataset.columns if "Ground" in col]
X = dataset.drop(columns=["Label"] + ground_cols)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset["Label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

model.save("exercise_classifier.h5")



converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("exercise_classifier.tflite", "wb") as f:
   f.write(tflite_model)

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="../../app/src/main/assets/exercise_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT:", input_details)
print("OUTPUT:", output_details)
