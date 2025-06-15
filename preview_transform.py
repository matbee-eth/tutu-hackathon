import cv2
import numpy as np
import os

# --- Configuration ---
CAMERA_INDICES = [0, 1]  # Camera 0 is source, Camera 1 is destination/reference
MATRIX_FILE = 'transformation_matrix.npy'
# ---------------------

def main():
    # Check if the transformation matrix exists
    if not os.path.exists(MATRIX_FILE):
        print(f"Error: Transformation matrix file not found at '{MATRIX_FILE}'")
        print("Please run the 'interactive_transform.py' script first to generate it.")
        return

    # Load the transformation matrix
    try:
        transformation_matrix = np.load(MATRIX_FILE)
        print(f"Successfully loaded transformation matrix from '{MATRIX_FILE}'.")
    except Exception as e:
        print(f"Error loading matrix file: {e}")
        return

    # Initialize cameras
    caps = {}
    for i in CAMERA_INDICES:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps[i] = cap
        else:
            print(f"Error: Camera {i} could not be opened.")
    
    if len(caps) != len(CAMERA_INDICES):
        print("Error: Could not open all required cameras. Exiting.")
        return

    # Get the frame size from the destination camera (cam 1) for the warp output
    ret, frame1 = caps[CAMERA_INDICES[1]].read()
    if not ret:
        print(f"Error: Could not read frame from reference camera {CAMERA_INDICES[1]}.")
        return
    h, w, _ = frame1.shape
    result_frame_size = (w, h)

    # Create windows
    cv2.namedWindow('Warped View (Cam 0 -> Cam 1 Perspective)')
    cv2.namedWindow('Original View (Cam 1)')
    
    print("\nDisplaying preview. Press 'q' to quit.")

    while True:
        # Read frames from both cameras
        ret0, frame0 = caps[CAMERA_INDICES[0]].read()
        ret1, frame1 = caps[CAMERA_INDICES[1]].read()

        if not ret0 or not ret1:
            print("Warning: Failed to capture frame from one or both cameras.")
            break
            
        # Warp the source camera's view (cam 0) using the matrix
        warped_frame = cv2.warpPerspective(frame0, transformation_matrix, result_frame_size)
        
        # Display the warped and original views
        cv2.imshow('Warped View (Cam 0 -> Cam 1 Perspective)', warped_frame)
        cv2.imshow('Original View (Cam 1)', frame1)

        # Allow overlaying the views for better comparison
        # You can adjust the alpha for more or less transparency
        blended_frame = cv2.addWeighted(frame1, 0.5, warped_frame, 0.5, 0)
        cv2.imshow('Blended View (For Alignment Check)', blended_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("Preview ended.")

if __name__ == "__main__":
    main() 