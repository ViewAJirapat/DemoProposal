import pyrealsense2 as rs
import os
import time

def main():
    # Define the directory where the script and the bag file will reside
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure the directory exists (OS-level check)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find the next available file number to avoid overwriting
    base_name = "outdoor_dataset"
    counter = 1
    while True:
        bag_name = f"{base_name}_{counter}.bag"
        bag_path = os.path.join(output_dir, bag_name)
        if not os.path.exists(bag_path):
            break
        counter += 1

    # Initialize the pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable recording to the bag file
    config.enable_record_to_file(bag_path)

    # Configure streams: Color, Depth, and Infrared (Left/Right)
    width, height, fps = 640, 480, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

    print(f"Initializing stream. Recording will be saved to: {bag_path}")

    try:
        # Start the pipeline
        pipeline_profile = pipeline.start(config)

        # Get the device and the depth sensor to configure hardware settings
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()

        # 1. Disable the IR emitter for outdoor environments
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            print("IR emitter disabled.")

        # 2. Enable auto-exposure
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("Auto-exposure enabled.")

        # 4. Apply an Align object to align depth frames to color frames
        align_to = rs.stream.color
        align = rs.align(align_to)

        print("Recording started. Press Ctrl+C to stop.")

        # Continuously record streams
        while True:
            # Wait for the next set of frames
            frames = pipeline.wait_for_frames()
            
            # Align the depth frames to the color frames
            # Even though we don't visualize it, processing it here ensures 
            # we are actively fetching and handling frames while the pipeline records.
            aligned_frames = align.process(frames)
            
            # Optional short sleep to prevent the loop from spinning too fast
            # time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nRecording interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn error occurred during recording: {e}")
    finally:
        # Cleanly stop the pipeline and release resources
        pipeline.stop()
        print(f"Pipeline stopped. Data successfully saved to {bag_path}")

if __name__ == "__main__":
    main()
