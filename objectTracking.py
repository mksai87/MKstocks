import cv2
import numpy as np
import firebase_admin
from firebase_admin import db, credentials

# Initialize Firebase
cred = credentials.Certificate("rec.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://stock-bdc3f-default-rtdb.asia-southeast1.firebasedatabase.app/"})
ref = db.reference("/MK")  # Change the reference to the "MK" child node

def track_orange_object():
    # Open the webcam (you can replace '0' with the camera index if you have multiple cameras)
    cap = cv2.VideoCapture("km_track_720p_30f_20240311_091942.mp4")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is not empty
        if not ret or frame is None:
            print("Error: Failed to read frame from webcam")
            break

        # Resize the frame to 300x250
        frame = cv2.resize(frame, (500, 300))

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the orange color
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([15, 255, 255])

        # Create a mask using the inRange function to filter out only the orange color
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, find the largest one
        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            # Get the centroid of the object
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Calculate the distance from the center
                distance_from_center = cx - frame.shape[1] // 2

                # Draw a circle around the object and the centroid
                cv2.drawContours(frame, [max_contour], -1, (0, 165, 255), 2)  # Orange color
                cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)  # Black centroid
                cv2.circle(frame, (cx, cy), 30, (0, 165, 255), 2)  # Orange circle

                # Display the distance from the center
                cv2.putText(frame, f"Distance from center: {distance_from_center}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text
                ref.update({"StockValue": distance_from_center})

        # Display the resulting frame
        cv2.imshow('Orange Object Tracker', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_orange_object()
