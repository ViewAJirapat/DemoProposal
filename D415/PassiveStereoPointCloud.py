import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

def main():
    # ==========================================
    # 1. ตั้งค่าการเชื่อมต่อและสตรีมมิ่งกล้อง D415
    # ==========================================
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    print("กำลังเชื่อมต่อกล้อง...")
    profile = pipeline.start(config)

    # ==========================================
    # 2. ตั้งค่าเซ็นเซอร์ (เปิด IR Emitter สำหรับการใช้งานในร่ม)
    # ==========================================
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        # ตั้งค่าเป็น 1 (Active Stereo) เพื่อสาด Pattern ลงไปแก้ปัญหาภาพแหว่งบนวัตถุผิวเรียบ/สีดำ
        # หากนำไปใช้วิ่งทดสอบกลางแจ้ง (Outdoor) ให้เปลี่ยนเป็น 0 (Passive)
        depth_sensor.set_option(rs.option.emitter_enabled, 0) 
        print("เปิดใช้งาน IR Emitter (Passive Stereo)")

    # ==========================================
    # 3. เตรียมตัวช่วยและ Filters ต่างๆ
    # ==========================================
    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    # ชุดคำสั่งกรองสัญญาณรบกวนและอุดรูรั่วของความลึก
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # ==========================================
    # 4. ตั้งค่าหน้าต่างแสดงผล 3 มิติ (Open3D)
    # ==========================================
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud (3D)", width=800, height=600)
    
    # ปรับแต่งให้มองง่ายขึ้น
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15]) # พื้นหลังสีเทาเข้ม
    opt.point_size = 4.0 # ขยายขนาดจุด

    o3d_pcd = o3d.geometry.PointCloud()
    is_initialized = False

    print("เริ่มการทำงาน! (กดปุ่ม 'q' หรือ 'ESC' ที่หน้าต่างภาพเพื่อออก)")

    try:
        while True:
            # รับเฟรมภาพจากกล้อง
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # นำ Depth Frame ไปผ่านกระบวนการปรับปรุงคุณภาพ (Filters)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            # ==========================================
            # 5. ประมวลผลและแสดงภาพ 2D (OpenCV)
            # ==========================================
            color_image = np.asanyarray(color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) # แปลงสีกลับให้ OpenCV แสดงผลถูก
            cv2.imshow("RGB Image (2D)", color_image_bgr)

            # ==========================================
            # 6. ประมวลผลและแสดง Point Cloud (Open3D)
            # ==========================================
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            
            # ดึงพิกัด (Vertices) และสี
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            colors = color_image.reshape(-1, 3) / 255.0 
            
            # อัปเดตข้อมูลให้ Open3D
            o3d_pcd.points = o3d.utility.Vector3dVector(vtx)
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

            # พลิกแกน (ปรับให้ตรงกับโลกความเป็นจริง)
            o3d_pcd.transform([[1, 0, 0, 0], 
                               [0, -1, 0, 0], 
                               [0, 0, -1, 0], 
                               [0, 0, 0, 1]])

            # เรนเดอร์ลงหน้าต่าง
            if not is_initialized:
                vis.add_geometry(o3d_pcd)
                # เพิ่มแกนพิกัด XYZ (แดง เขียว น้ำเงิน) ช่วยกะทิศทาง
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                vis.add_geometry(axis)
                is_initialized = True
            else:
                vis.update_geometry(o3d_pcd)
            
            vis.poll_events()
            vis.update_renderer()

            # ตรวจจับการกดคีย์บอร์ดที่หน้าต่าง OpenCV
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
    finally:
        print("กำลังปิดโปรแกรม...")
        pipeline.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()

if __name__ == "__main__":
    main()