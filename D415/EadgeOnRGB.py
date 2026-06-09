import pyrealsense2 as rs
import numpy as np
import cv2

# ตั้งค่า Pipeline สำหรับสตรีมมิ่งกล้อง D415
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    print("กำลังสตรีมภาพ... กด 'q' เพื่อออก")
    while True:
        # รอรับเฟรมภาพ
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # แปลงข้อมูลเฟรมเป็น Numpy Array
        color_image = np.asanyarray(color_frame.get_data())

        # กระบวนการประมวลผลภาพ
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # --- ส่วนที่เพิ่มเข้ามาสำหรับการรวมภาพหน้าต่างเดียว ---
        
        # 1. แปลงภาพขอบ (1 Channel) ให้เป็น 3 Channels (BGR เทียม)
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 2. นำอาร์เรย์ภาพมาต่อกันในแนวนอน (ซ้าย: ต้นฉบับ, ขวา: ภาพขอบ)
        # หมายเหตุ: หากต้องการต่อแนวตั้ง (บน-ล่าง) ให้เปลี่ยนเป็น np.vstack()
        combined_image = np.hstack((color_image, edges_3channel))

        # แสดงผลลัพธ์ในหน้าต่างเดียว
        cv2.imshow('Intel D415: Original vs Edge Detection', combined_image)

        # กด 'q' เพื่อหยุดการทำงาน
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()