import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# 1. ตั้งค่าการอ่านไฟล์ .bag
# ==========================================
pipeline = rs.pipeline()
config = rs.config()

import os

# 💡 ใส่ชื่อไฟล์ .bag ของคุณ
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
#bag_file_path = os.path.join(data_dir, "dataset_10sec.bag")
#bag_file_path = os.path.join(data_dir, "first_round.bag")
bag_file_path = os.path.join(data_dir, "sec.bag")

rs.config.enable_device_from_file(config, bag_file_path, repeat_playback=True)

print(f"กำลังเปิดอ่านไฟล์: {bag_file_path}")
profile = pipeline.start(config)

playback = profile.get_device().as_playback()
playback.set_real_time(True)

print("เริ่มแสดงผล: กด 'q' หรือ 'Esc' เพื่อปิดหน้าต่าง")

try:
    while True:
        try:
            # ดึงเฟรมทั้งหมดที่มี (โดยไม่ทำการ Align)
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            continue
            
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame and not color_frame:
            continue

        # -----------------------------------------------------
        # กรณีที่ 1: มีข้อมูลครบทั้ง 2 อย่าง -> นำมาต่อกันในหน้าต่างเดียว
        # -----------------------------------------------------
        if depth_frame and color_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # ปรับสีภาพ Depth
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 💡 ป้องกันบั๊ก: ตรวจสอบว่าความสูงของภาพเท่ากันหรือไม่ก่อนนำมาต่อกัน
            if depth_colormap.shape[0] != color_image.shape[0] or depth_colormap.shape[1] != color_image.shape[1]:
                # ถ้าไม่เท่า ให้ปรับขนาดภาพ Depth ให้เท่ากับ RGB
                depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

            # นำภาพมาต่อกันในแนวนอน (ซ้าย: RGB, ขวา: Depth)
            combined_image = np.hstack((color_image, depth_colormap))
            
            cv2.imshow('RGB and Depth (Combined)', combined_image)

        # -----------------------------------------------------
        # กรณีที่ 2: มีแค่ภาพ Depth อย่างเดียว
        # -----------------------------------------------------
        elif depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Depth Only', depth_colormap)

        # -----------------------------------------------------
        # กรณีที่ 3: มีแค่ภาพ RGB อย่างเดียว
        # -----------------------------------------------------
        elif color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RGB Only', color_image)

        # เช็คการกดปุ่มเพื่อออก
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            print("ปิดการแสดงผล...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()