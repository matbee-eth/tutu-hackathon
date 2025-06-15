import cv2
import numpy as np
import os
import argparse
from glob import glob
import sys

def create_mask_overlay(base_frame, warped_overlay):
    """
    Creates an opaque overlay using a mask from a pre-warped frame.

    Args:
        base_frame: The base image (numpy array).
        warped_overlay: The already-warped overlay image (numpy array).

    Returns:
        The combined image with the overlay (numpy array).
    """
    # Create a mask of the warped overlay.
    # The mask is black where the warped overlay is black.
    gray_warped = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
    # The original threshold was 1, which was too high for the dark thermal image.
    # Changing to 0 means any non-black pixel in the warped source will be part of the mask.
    _, mask = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY)
    
    # Invert the mask to get the area of the base frame that should be visible
    mask_inv = cv2.bitwise_not(mask)
    
    # Black-out the area of the overlay in the base frame
    base_bg = cv2.bitwise_and(base_frame, base_frame, mask=mask_inv)
    
    # Take only the region of the overlay from the warped overlay image.
    overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)
    
    # Add the foreground overlay to the background base frame
    combined_frame = cv2.add(base_bg, overlay_fg)
    
    return combined_frame

def process_and_save_video(source_video_path, dest_video_path, output_video_path, transformation_matrix, method, alpha, beta, debug=False, debug_frames=10, draw_border=False):
    """
    Processes a pair of videos and saves the overlaid result.
    """
    source_cap = cv2.VideoCapture(source_video_path)
    dest_cap = cv2.VideoCapture(dest_video_path)

    if not source_cap.isOpened() or not dest_cap.isOpened():
        print(f"\nError: Could not open one or both video files: {source_video_path}, {dest_video_path}")
        if source_cap.isOpened(): source_cap.release()
        if dest_cap.isOpened(): dest_cap.release()
        return

    # Get properties from the destination video for the output
    width = int(dest_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(dest_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = dest_cap.get(cv2.CAP_PROP_FPS)
    
    total_frames = min(
        int(dest_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    
    frames_to_process = total_frames
    if debug:
        print("\n--- DEBUG MODE ENABLED FOR VIDEO ---")
        frames_to_process = min(total_frames, debug_frames)
        print(f"Processing {frames_to_process} frames for debug.")

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"\nProcessing '{os.path.basename(source_video_path)}':")
    print(f" -> Output: '{output_video_path}'")

    processed_frames = 0
    while processed_frames < frames_to_process:
        ret_source, source_frame = source_cap.read()
        ret_dest, dest_frame = dest_cap.read()

        if not ret_source or not ret_dest:
            break
        
        # --- Warp and Combine ---
        h, w, _ = dest_frame.shape
        warped_overlay = cv2.warpPerspective(source_frame, transformation_matrix, (w, h))

        if method == 'blend':
            combined_frame = cv2.addWeighted(dest_frame, alpha, warped_overlay, beta, 0.0)
        else: # 'mask'
            combined_frame = create_mask_overlay(dest_frame, warped_overlay)
        
        # --- Draw Border if requested ---
        if draw_border:
            # Get the corners of the source frame
            src_h, src_w, _ = source_frame.shape
            src_corners = np.array([
                [0, 0],         # Top-left
                [src_w-1, 0],   # Top-right
                [src_w-1, src_h-1], # Bottom-right
                [0, src_h-1]    # Bottom-left
            ], dtype='float32')
            
            # Reshape for the perspective transform function
            src_corners = np.array([src_corners])
            
            # Apply the perspective transformation to the corners
            dst_corners = cv2.perspectiveTransform(src_corners, transformation_matrix)
            
            # Draw lines between the transformed corners
            # The points need to be integers for drawing
            border_points = np.int32(dst_corners[0])
            cv2.polylines(combined_frame, [border_points], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        if debug:
            # Save intermediate images for debugging
            debug_dir = os.path.join(os.path.dirname(output_video_path), 'debug_output')
            os.makedirs(debug_dir, exist_ok=True)
            if processed_frames == 0:
                print(f"Saving debug images to '{os.path.abspath(debug_dir)}'")

            frame_num_str = f"{processed_frames:04d}"
            cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_01_dest_frame.png'), dest_frame)
            cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_02_source_frame.png'), source_frame)
            cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_03_warped_source.png'), warped_overlay)

            if method == 'mask':
                gray_warped = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_04_mask.png'), mask)

                mask_inv = cv2.bitwise_not(mask)
                base_bg = cv2.bitwise_and(dest_frame, dest_frame, mask=mask_inv)
                overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

                cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_05_base_bg_masked.png'), base_bg)
                cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_06_overlay_fg_masked.png'), overlay_fg)
            
            cv2.imwrite(os.path.join(debug_dir, f'frame_{frame_num_str}_07_combined_frame.png'), combined_frame)
        else:
            out.write(combined_frame)
        
        processed_frames += 1
        print(f"\r  - Frame {processed_frames}/{frames_to_process}", end="")

    if not debug:
        out.release()

    source_cap.release()
    dest_cap.release()
    
    if debug:
        print(f"\nDebug mode finished. {processed_frames} frames saved as images. No video was written.")
        sys.exit(0)

    print("... Done.")


def main():
    """
    Main function to parse arguments and run the video overlay process on a LeRobot dataset.
    """
    parser = argparse.ArgumentParser(
        description="Create overlay videos for a LeRobot dataset from two camera views."
    )
    parser.add_argument('--dataset-path', required=True, help="Path to the root of the LeRobot dataset.")
    parser.add_argument('--source-camera', required=True, help="Name of the source camera view (e.g., 'thermal_camera'). This view will be warped.")
    parser.add_argument('--destination-camera', required=True, help="Name of the destination camera view (e.g., 'rgb_camera'). This is the base view.")
    parser.add_argument('--output-camera-name', default='merged_camera', help="Name for the new camera view directory containing the overlaid videos.")
    parser.add_argument('--matrix', default='transformation_matrix.npy', help="Path to the transformation matrix file (.npy).")
    parser.add_argument('--method', default='blend', choices=['blend', 'mask'], help="The overlay method to use: 'blend' (transparent) or 'mask' (opaque).")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for the destination (base) video in blend mode. Used only with --method=blend.")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight for the source (overlay) video in blend mode. Used only with --method=blend.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode to process only one frame and save intermediate images.")
    parser.add_argument('--debug-frames', type=int, default=10, help="Number of frames to process in debug mode. Only used with --debug.")
    parser.add_argument('--invert-matrix', action='store_true', help="Invert the transformation matrix before applying it. Use if the overlay is not visible.")
    parser.add_argument('--draw-border', action='store_true', help="Draw a border around the warped source image on the output video.")
    parser.add_argument('--source-camera-index', type=int, default=0, choices=[0, 1], help="The index of the source camera (0 or 1) used during transformation matrix generation. Defaults to 0.")
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset path not found at '{args.dataset_path}'")
        return
    if not os.path.exists(args.matrix):
        print(f"Error: Transformation matrix not found at '{args.matrix}'")
        print("Please run 'interactive_transform.py' to generate it.")
        return

    # --- Load Transformation Matrix ---
    try:
        transformation_matrix = np.load(args.matrix)
        if (args.source_camera_index == 1 and not args.invert_matrix) or \
           (args.source_camera_index == 0 and args.invert_matrix):
            print("Inverting the transformation matrix because the source camera is not at index 0 or --invert-matrix is explicitly used.")
            transformation_matrix = np.linalg.inv(transformation_matrix)
        
        print("Transformation matrix loaded successfully.")
        if args.debug:
            print("--- Transformation Matrix (after potential inversion) ---")
            print(transformation_matrix)
            print("-----------------------------------------------------")
    except Exception as e:
        print(f"Error loading or inverting matrix file: {e}")
        return

    # --- Find and Process Videos ---
    videos_root_path = os.path.join(args.dataset_path, 'videos')
    if not os.path.isdir(videos_root_path):
        print(f"Error: 'videos' directory not found in dataset path: '{videos_root_path}'")
        return
    
    # Find all episode videos in the destination camera directory
    search_pattern = os.path.join(
        videos_root_path,
        '**/observation.images.{}/episode_*.mp4'.format(args.destination_camera)
    )
    destination_videos = glob(search_pattern, recursive=True)

    if not destination_videos:
        print(f"Error: No videos found for the destination camera '{args.destination_camera}' in the dataset.")
        return
    
    print(f"Found {len(destination_videos)} videos to process for camera '{args.destination_camera}'.")

    for dest_video_path in destination_videos:
        # Construct the corresponding source and output paths
        source_video_path = dest_video_path.replace(
            f'observation.images.{args.destination_camera}',
            f'observation.images.{args.source_camera}'
        )
        output_video_path = dest_video_path.replace(
            f'observation.images.{args.destination_camera}',
            f'observation.images.{args.output_camera_name}'
        )
        
        # Check if the corresponding source video exists
        if not os.path.exists(source_video_path):
            print(f"\nWarning: Source video not found for '{dest_video_path}', skipping.")
            print(f"  - Looked for: '{source_video_path}'")
            continue

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # Process the video pair
        process_and_save_video(source_video_path, dest_video_path, output_video_path, transformation_matrix, method=args.method, alpha=args.alpha, beta=args.beta, debug=args.debug, debug_frames=args.debug_frames, draw_border=args.draw_border)

    print("\n--- All videos processed. ---")


if __name__ == "__main__":
    main()