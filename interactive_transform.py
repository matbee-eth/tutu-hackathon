import cv2
import numpy as np
import os

# --- Configuration ---
CAMERA_INDICES = [0, 1]
WINDOW_NAMES = [f'Camera {i}' for i in CAMERA_INDICES]
NUM_POINTS = 4
OUTPUT_MATRIX_FILE = 'transformation_matrix.npy'
# ---------------------

# Global variables to store points
points = {name: [] for name in WINDOW_NAMES}
is_capturing_points = True

def select_point(event, x, y, flags, param):
    """Mouse callback function to select points."""
    global points, is_capturing_points
    if not is_capturing_points:
        return

    window_name = param['name']
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points[window_name]) < NUM_POINTS:
            points[window_name].append((x, y))
            print(f"Point {len(points[window_name])}/{NUM_POINTS} selected for {window_name} at ({x}, {y})")

def draw_points(frame, selected_points):
    """Draws selected points and connecting lines on the frame."""
    for i, point in enumerate(selected_points):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i+1), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if len(selected_points) > 1:
        cv2.polylines(frame, [np.array(selected_points)], isClosed=False, color=(0, 255, 0), thickness=2)

def draw_instructions(frame, window_name, num_selected):
    """Draws instructions on the frame."""
    texts = [
        f"Click to select {NUM_POINTS} points for {window_name}.",
        f"Points selected: {num_selected}/{NUM_POINTS}",
        "Press 'r' to reset points for this window.",
        "Once all points are selected for both windows,",
        "the transformation will be calculated.",
        "Press 'q' to quit."
    ]
    y0, dy = 30, 30
    for i, line in enumerate(texts):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    global points, is_capturing_points

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

    # Create windows and set mouse callbacks
    for name in WINDOW_NAMES:
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, select_point, {'name': name})

    print("--- Interactive Point Selection ---")
    print(f"Please select {NUM_POINTS} corresponding points in each window.")
    print("Press 'r' to reset points for the active window.")
    print("Press 'q' to quit.")

    transformation_matrix = None
    result_frame_size = None

    while True:
        frames = {}
        all_frames_read = True
        for i, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                frames[i] = frame.copy()
                if result_frame_size is None:
                    # Use the first camera's resolution for the output
                    h, w, _ = frame.shape
                    result_frame_size = (w, h)
            else:
                print(f"Warning: Failed to capture frame from camera {i}.")
                all_frames_read = False
        
        if not all_frames_read:
            break

        # Display frames and draw points/instructions
        for i, name in zip(CAMERA_INDICES, WINDOW_NAMES):
            frame_copy = frames[i].copy()
            if is_capturing_points:
                draw_instructions(frame_copy, name, len(points[name]))
            draw_points(frame_copy, points[name])
            cv2.imshow(name, frame_copy)

        # Check if we have enough points to calculate the transformation
        if is_capturing_points and all(len(p) == NUM_POINTS for p in points.values()):
            print("\nAll points selected. Calculating transformation...")
            # Points from Camera 0 (source) and Camera 1 (destination)
            src_pts = np.array(points[WINDOW_NAMES[0]], dtype='float32')
            dst_pts = np.array(points[WINDOW_NAMES[1]], dtype='float32')
            
            # Calculate the perspective transformation matrix
            transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if transformation_matrix is not None:
                print("Transformation matrix calculated successfully.")
                np.save(OUTPUT_MATRIX_FILE, transformation_matrix)
                print(f"Matrix saved to '{os.path.abspath(OUTPUT_MATRIX_FILE)}'")
                is_capturing_points = False # Stop capturing points
                # Close selection windows and open result window
                for name in WINDOW_NAMES:
                    cv2.destroyWindow(name)
                cv2.namedWindow('Warped Result')
            else:
                print("Error: Could not find a valid transformation. Please reset and try again.")
                # Allow user to reset and re-select
                points = {name: [] for name in WINDOW_NAMES}


        if not is_capturing_points and transformation_matrix is not None:
            # Warp the first camera's view using the matrix
            warped_frame = cv2.warpPerspective(frames[CAMERA_INDICES[0]], transformation_matrix, result_frame_size)
            
            # Display the original and warped views
            cv2.imshow('Warped Result', warped_frame)
            cv2.imshow(f'Original {WINDOW_NAMES[1]}', frames[CAMERA_INDICES[1]])
            
            cv2.putText(warped_frame, "Press 'r' to re-select points.", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2,cv2.LINE_AA)
            cv2.putText(warped_frame, "Press 'q' to quit.", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2,cv2.LINE_AA)
            cv2.imshow('Warped Result', warped_frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if is_capturing_points:
                 # Find which window is focused and reset its points
                 # This is a bit of a hack as OpenCV doesn't have a direct way to get focused window
                 # We assume the user will have their mouse over the window they want to reset.
                 # A more robust solution might require a different UI approach.
                 print("Resetting points. Please re-select for all windows.")

            points = {name: [] for name in WINDOW_NAMES}
            is_capturing_points = True
            transformation_matrix = None
            print("\nPoints reset. Please select new points.")
            # Recreate windows if they were closed
            cv2.destroyAllWindows()
            for name in WINDOW_NAMES:
                cv2.namedWindow(name)
                cv2.setMouseCallback(name, select_point, {'name': name})


    # Cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")

if __name__ == "__main__":
    main() 