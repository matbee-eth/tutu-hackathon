import cv2
import numpy as np
import os
import time

def create_pixelated_grid(frame, pixel_size=5):
    """
    Creates a pixelated version of a frame by averaging colors in square blocks.

    Args:
        frame: The input image/frame (numpy array).
        pixel_size: The size of the pixel blocks.

    Returns:
        The pixelated image (numpy array).
    """
    h, w, _ = frame.shape
    pixelated_image = np.zeros_like(frame)

    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            # Define the region of interest (ROI)
            y_end = min(y + pixel_size, h)
            x_end = min(x + pixel_size, w)
            roi = frame[y:y_end, x:x_end]

            if roi.size > 0:
                # Calculate the average color of the ROI
                avg_color = np.mean(roi, axis=(0, 1))
                # Convert color to a tuple of integers for OpenCV
                color = tuple(int(c) for c in avg_color)
                # Draw a filled rectangle with the average color
                cv2.rectangle(pixelated_image, (x, y), (x_end, y_end), color, -1)

    return pixelated_image

def main():
    # --- Configuration ---
    CAMERA_INDICES = [0, 1]
    NUM_FRAMES = 10
    PIXEL_SIZE = 5
    OUTPUT_DIR = "outputs/pixelated_frames"
    # --- End Configuration ---

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: '{os.path.abspath(OUTPUT_DIR)}'")

    # Initialize cameras
    caps = {}
    for i in CAMERA_INDICES:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps[i] = cap
        else:
            print(f"Warning: Camera {i} could not be opened.")
    
    if not caps:
        print("Error: No cameras available. Exiting.")
        return

    print(f"\nCapturing {NUM_FRAMES} frames from cameras: {list(caps.keys())}")

    for frame_num in range(NUM_FRAMES):
        print(f"--- Frame {frame_num + 1}/{NUM_FRAMES} ---")
        for cam_index, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                print(f"  - Processing frame from camera {cam_index}...")
                pixelated_frame = create_pixelated_grid(frame, PIXEL_SIZE)
                
                filename = f"cam_{cam_index}_frame_{frame_num:02d}.png"
                output_path = os.path.join(OUTPUT_DIR, filename)
                
                cv2.imwrite(output_path, pixelated_frame)
                print(f"    - Saved to {output_path}")
            else:
                print(f"  - Warning: Failed to capture frame from camera {cam_index}.")
        
        # Small delay to allow cameras to be ready for next frame
        if frame_num < NUM_FRAMES - 1:
            time.sleep(0.1)

    # Release camera resources
    for cap in caps.values():
        cap.release()
    
    print("\nCapture and pixelation complete.")

if __name__ == "__main__":
    main() 