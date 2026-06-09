import pyrealsense2 as rs
import time

# 1. กำหนดค่า Pipeline
pipeline = rs.pipeline()
config = rs.config()

# ตั้งค่าความละเอียดเฟรม
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 2. 💡 คำสั่งสำคัญ: สั่งให้บันทึกเป็นไฟล์ .bag
bag_filename = "obstacle_dataset_windows.bag"
config.enable_record_to_file(bag_filename)

print("กำลังเชื่อมต่อกล้อง และเริ่มบันทึกไฟล์ .bag...")
profile = pipeline.start(config)

# ปิด IR Emitter เพื่อให้เป็น Passive Stereo
device = profile.get_device()
depth_sensor = device.query_sensors()[0]
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)

try:
    print(f"กำลังบันทึกข้อมูลลงไฟล์: {bag_filename}")
    print("กด 'Ctrl+C' ใน Terminal เพื่อหยุดการบันทึก")
    
    # ปล่อยให้กล้องรันและเซฟข้อมูลไปเรื่อยๆ (ไม่มีการแสดงหน้าต่างภาพเพื่อลดภาระเครื่อง)
    while True:
        pipeline.wait_for_frames()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nหยุดการบันทึกเรียบร้อยแล้ว!")

finally:
    pipeline.stop()