import os
import cv2
import numpy as np
import tensorflow as tf

def tflite_detect_images(model_path, images_directory, label_map_path, min_conf=0.87):
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

    # Iterate over all images in the directory
    for filename in os.listdir(images_directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Load image and resize to expected shape [1xHxWx3]
            image_path = os.path.join(images_directory, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values if using a floating model
            input_mean = 127.5
            input_std = 127.5
            input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
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
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    # Draw bounding box on the image
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Put label text on the image
                    label = f"{labels[int(classes[i])]}: {scores[i]:.2f}"
                    cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the image with bounding boxes
            cv2.imshow("Object Detection", image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

# Example usage
model_path = "Kadett/custom_model_lite/detect.tflite"
images_directory = "TestImages"
label_map_path = "Kadett/custom_model_lite/labelmap.txt"
tflite_detect_images(model_path, images_directory, label_map_path)
