import cv2

# Load the image
img = cv2.imread("image.jpg")

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")

# Preprocess the image
blob = cv2.dnn.blobFromImage(img, 1.0, (224,224), (104,117,123))

# Run object detection
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(detections.shape[2]):
    # Extract the confidence score
    confidence = detections[0,0,i,2]

    # Filter out weak detections
    if confidence > 0.5:
        # Compute the bounding box coordinates
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box on the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (255,0,0), 2)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
