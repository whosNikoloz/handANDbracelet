import os
import cv2
import numpy as np
import tensorflow as tf

def tflite_detect_camera(model_path, label_map_path, min_conf=0.8):
    # Load the label map into memory
    with open(label_map_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model into memory
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if you have multiple cameras

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame from camera")
            break

        # Resize frame to expected shape [1xHxWx3]
        frame_resized = cv2.resize(frame, (width, height))

        # Normalize pixel values if using a floating model
        input_mean = 127.5
        input_std = 127.5
        input_data = (np.float32(frame_resized) - input_mean) / input_std

        # Expand dimensions to have shape [1xHxWx3]
        input_data = np.expand_dims(input_data, axis=0)

        # Perform the actual detection by running the model with the frame as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Get bounding box coordinates
                ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
                xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
                ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
                xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))

                # Draw bounding box on the frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Put label text on the frame
                label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Example usage
model_path = "handANDbracelete/custom_model_lite/detect.tflite"
label_map_path = "handANDbracelete/custom_model_lite/labelmap.txt"
tflite_detect_camera(model_path, label_map_path)
