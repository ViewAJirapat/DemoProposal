import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Hardware Setup: Enable Color and Infrared streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Access the depth sensor to explicitly disable the IR emitter
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        print("IR Emitter has been successfully disabled.")

    print("Streaming started... Press the 'q' key on the video window to quit.")

    try:
        while True:
            # Wait for a coherent set of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1) # Get the left infrared stream

            if not color_frame or not ir_frame:
                continue

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Define target dimensions
            dim = (640, 480)

            # 1. RGB Image: The original color frame (loaded as BGR by default in OpenCV)
            rgb_img = cv2.resize(color_image, dim)

            # 2. IR Image: Scale, resize, and convert to an 8-bit 3-channel format
            ir_resized = cv2.resize(ir_image, dim)
            ir_3_channel = cv2.cvtColor(ir_resized, cv2.COLOR_GRAY2BGR)

            # 3. Grayscale Image: Convert RGB to Gray, then back to 3-channel BGR for stacking
            gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            gray_3_channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            # 4. Red Channel (Set B and G channels to 0)
            red_channel = rgb_img.copy()
            red_channel[:, :, 0] = 0 # Blue = 0
            red_channel[:, :, 1] = 0 # Green = 0

            # 5. Green Channel (Set B and R channels to 0)
            green_channel = rgb_img.copy()
            green_channel[:, :, 0] = 0 # Blue = 0
            green_channel[:, :, 2] = 0 # Red = 0

            # 6. Blue Channel (Set G and R channels to 0)
            blue_channel = rgb_img.copy()
            blue_channel[:, :, 1] = 0 # Green = 0
            blue_channel[:, :, 2] = 0 # Red = 0

            # --- Add Text Labels to Each Frame ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255) # White text
            thickness = 2
            position = (15, 35) # Top-left corner (x, y)

            cv2.putText(rgb_img, 'RGB Image', position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(ir_3_channel, 'IR Image', position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(gray_3_channel, 'Grayscale Image', position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(red_channel, 'Red Channel', position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(green_channel, 'Green Channel', position, font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(blue_channel, 'Blue Channel', position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Display Layout: 2x3 Grid
            # Top row: RGB, IR, Grayscale
            row_1 = np.hstack((rgb_img, ir_3_channel, gray_3_channel))
            
            # Bottom row: Red, Green, Blue
            row_2 = np.hstack((red_channel, green_channel, blue_channel))

            # Combine both rows vertically
            combined_grid = np.vstack((row_1, row_2))

            # Display the combined window
            cv2.namedWindow('RealSense Stream (6 Variations)', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Stream (6 Variations)', combined_grid)

            # Wait for key press and exit properly if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Properly stop the pipeline and close all OpenCV windows
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")

if __name__ == "__main__":
    main()
