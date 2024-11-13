import cv2
from ultralytics import YOLO

# Import the trained model
model = YOLO('best.pt')  

# OPen the camera
cap = cv2.VideoCapture(1)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Spoon detection of the frame
    results = model(frame, conf = 0.65)

    # Show results on the screen
    annotated_frame = results[0].plot()
    cv2.imshow('Detección de Cucharas', annotated_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
