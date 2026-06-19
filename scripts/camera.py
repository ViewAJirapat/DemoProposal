import pyrealsense2 as rs
import time

def record_realsense(filename="realsense_record.bag", duration=10):
    # 1. ตั้งค่า Pipeline และ Config
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. เปิดใช้งาน Stream ทั้ง Depth และ RGB (ความละเอียด 640x480 ที่ 30 FPS)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 3. สั่งให้บันทึกข้อมูลทั้งหมดลงไฟล์ .bag ตามชื่อที่กำหนด
    config.enable_record_to_file(filename)

    print(f"🎥 กำลังเริ่มเปิดกล้องและบันทึกข้อมูลลง '{filename}'...")
    
    # เริ่มต้นการสตรีม (และการบันทึกจะเริ่มอัตโนมัติ)
    profile = pipeline.start(config)
    
    # ปิดการทำงานของ Laser Emitter ชั่วคราว (ถ้าต้องการ) หรือปล่อยให้ทำงานปกติ
    # device = profile.get_device()
    # depth_sensor = device.first_depth_sensor()
    # depth_sensor.set_option(rs.option.emitter_enabled, 1)

    try:
        start_time = time.time()
        frame_count = 0
        
        print(f"⏱️ เริ่มจับเวลา {duration} วินาที กรุณาอย่าปิดโปรแกรม...")
        
        # 4. ลูปเพื่อดึงข้อมูลเฟรมไปเรื่อยๆ ตามเวลาที่กำหนด
        while (time.time() - start_time) < duration:
            # ต้องเรียก wait_for_frames() เพื่อให้กล้องอัปเดตและบันทึกข้อมูลลงไฟล์
            frames = pipeline.wait_for_frames()
            frame_count += 1
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        
    finally:
        # 5. ปิดกล้องและการบันทึกอย่างปลอดภัย
        pipeline.stop()
        print(f"✅ บันทึกเสร็จสิ้น! ข้อมูลทั้งหมดถูกเซฟเรียบร้อย (บันทึกไป {frame_count} เฟรม)")

import os

if __name__ == "__main__":
    # เรียกใช้งานฟังก์ชัน: ระบุชื่อไฟล์และเวลา (วินาที) ที่ต้องการ
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    record_realsense(os.path.join(data_dir, "dataset_10sec.bag"), 10)