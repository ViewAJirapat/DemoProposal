import pyrealsense2 as rs
import numpy as np
import cv2

def play_realsense_bag(filename="dataset_10sec.bag"):
    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, filename, repeat_playback=True)
    print(f"🎬 กำลังเล่นไฟล์: {filename}")
    
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 🌟 1. ตั้งค่า Colorizer ให้ล็อกระยะไกลสุดที่ 3 เมตร (3.0) เพื่อไม่ให้สีกระพริบ
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.max_distance, 3.0) 

    # 🌟 2. ประกาศใช้งาน Post-Processing Filters เหมือนในสคริปต์ก่อนหน้า
    spatial = rs.spatial_filter()    
    temporal = rs.temporal_filter()  
    hole_filling = rs.hole_filling_filter()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 🌟 3. นำข้อมูล Depth เฟรมมาผ่านฟิลเตอร์ทำความสะอาดก่อน
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # 4. ดึงข้อมูลภาพสี และใส่สีให้ Depth Map
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            images_side_by_side = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('RealSense Playback (RGB + Filtered Depth)', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Playback (RGB + Filtered Depth)', images_side_by_side)

            key = cv2.waitKey(30)
            if key & 0xFF == ord('q') or key == 27:
                print("🛑 หยุดการเล่นวิดีโอ")
                break

    except RuntimeError:
        print("จบไฟล์วิดีโอ")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    play_realsense_bag("dataset_10sec.bag")