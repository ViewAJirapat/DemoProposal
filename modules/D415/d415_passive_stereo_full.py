import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# ==========================================
# 1. การตั้งค่ากล้อง Intel RealSense D415
# ==========================================
pipeline = rs.pipeline()
config = rs.config()

# กำหนดความละเอียดและ Frame Rate (แนะนำ 640x480 สำหรับการประมวลผลแบบ Real-time)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# เริ่มการเชื่อมต่อกล้อง
profile = pipeline.start(config)

# ปิด IR Projector เพื่อใช้งานโหมด Passive Stereo (คำนวณความลึกจากแสงธรรมชาติ)
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    print("ระบบ: ปิด IR Projector (ทำงานในโหมด Passive Stereo)")

# สร้างออบเจ็กต์สำหรับ Align ภาพ Depth ให้ทาบตรงกับภาพ RGB
align_to = rs.stream.color
align = rs.align(align_to)

# ดึงค่าพารามิเตอร์เลนส์ (Intrinsics) เพื่อใช้แปลงพิกัด 2D เป็น 3D ให้ได้สัดส่วนจริง
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
)

# ==========================================
# 2. การตั้งค่าหน้าต่าง 3D (Open3D)
# ==========================================
vis = o3d.visualization.Visualizer()
vis.create_window("3D Point Cloud (Depth Colored | Normals | Axis)", width=800, height=600)

pcd = o3d.geometry.PointCloud()

# สร้างแกนพิกัด (Coordinate Frame) ขนาด 0.5 เมตร ที่จุดกำเนิด (0,0,0) ของกล้อง
# สีแดง = X, สีเขียว = Y, สีน้ำเงิน = Z
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

is_first_frame = True

# ==========================================
# 3. ลูปการประมวลผลหลัก (Main Loop)
# ==========================================
try:
    print("กำลังสตรีมภาพ... กด 'q' ที่หน้าต่าง OpenCV (2D) เพื่อออกจากโปรแกรม")
    print("เคล็ดลับ: ในหน้าต่าง 3D สามารถกดปุ่ม 'N' บนคีย์บอร์ดเพื่อเปิด/ปิดเส้นแสดงทิศทางพื้นผิว (Normals) ได้")
    
    while True:
        # รอรับเฟรมภาพและจับคู่ Depth กับ Color ให้ตรงกัน
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # แปลงข้อมูลเฟรมเป็น Numpy Array
        depth_data = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # ------------------------------------------
        # ส่วนที่ 3.1: การประมวลผลภาพ 2D และจัดหน้าต่างแบบ 2x2 Grid
        # ------------------------------------------
        
        # ก) สร้างภาพ Depth Colormap
        depth_normalized = cv2.normalize(depth_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap_2d = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # ข) สกัดขอบวัตถุ (Canny Edge Detection)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # ค) สร้างภาพซ้อนทับ (Overlay) เปลี่ยนเส้นขอบให้เป็นสีเขียวบนภาพจริง
        overlay_image = color_image.copy()
        overlay_image[edges == 255] = [0, 255, 0]

        # ง) นำภาพมาต่อกันเป็น 2x2 Grid
        top_row = np.hstack((color_image, depth_colormap_2d))      # บน: RGB | Depth
        bottom_row = np.hstack((edges_3channel, overlay_image))    # ล่าง: Edges | Overlay
        grid_2x2 = np.vstack((top_row, bottom_row))

        # ลดขนาดหน้าต่างลงครึ่งหนึ่ง (เพื่อให้ดูบนหน้าจอเดียวกับ 3D ได้ง่ายขึ้น)
        grid_2x2_resized = cv2.resize(grid_2x2, (640, 480))

        # จ) เพิ่มข้อความอธิบายทิศทางแกน (Legend) บนหน้าต่าง 2D
        legend_texts = [
            "Camera Optical Frame:",
            "- Z (Blue)  : Forward (Depth)",
            "- X (Red)   : Right",
            "- Y (Green) : Down"
        ]
        
        y0, dy = 30, 25 
        for i, text in enumerate(legend_texts):
            y = y0 + i * dy
            # วาดเส้นขอบสีดำ
            cv2.putText(grid_2x2_resized, text, (15, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            # วาดตัวหนังสือสีขาวทับ
            cv2.putText(grid_2x2_resized, text, (15, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('2x2 Grid: RGB | Depth | Edges | Overlay', grid_2x2_resized)

        # ------------------------------------------
        # ส่วนที่ 3.2: การสร้างและการแสดงผล 3D Point Cloud
        # ------------------------------------------
        
        # แปลงสี Depth Colormap จาก BGR (OpenCV) เป็น RGB (Open3D)
        depth_colormap_rgb = cv2.cvtColor(depth_colormap_2d, cv2.COLOR_BGR2RGB)
        
        # สร้างออบเจ็กต์ภาพสำหรับ Open3D (ใช้ภาพ Depth สี แทนภาพความจริง)
        o3d_color = o3d.geometry.Image(depth_colormap_rgb)
        o3d_depth = o3d.geometry.Image(depth_data)
        
        # สร้าง RGBD Image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, convert_rgb_to_intensity=False)
        
        # คำนวณ Point Cloud 
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic)
        
        # พลิกแกนภาพ (เพื่อไม่ให้ Point Cloud กลับหัว)
        temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # คำนวณทิศทางของพื้นผิว (Surface Normals) ด้วยรัศมี 10 cm
        temp_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # บังคับให้ Normals ทั้งหมดพุ่งเข้าหากล้อง
        temp_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        # อัปเดตข้อมูลพิกัด สี และทิศทางพื้นผิว ลงใน pcd หลักที่ใช้แสดงผล
        pcd.points = temp_pcd.points
        pcd.colors = temp_pcd.colors
        pcd.normals = temp_pcd.normals

        # เรนเดอร์กราฟิก 3D
        if is_first_frame:
            vis.add_geometry(pcd)
            vis.add_geometry(coordinate_frame) # เพิ่มแกนพิกัดเข้าไปในหน้าต่างเฉพาะเฟรมแรก
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()

        # กด 'q' ที่หน้าต่าง OpenCV เพื่อหยุดการทำงาน
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # คืนค่าทรัพยากรเมื่อจบการทำงาน
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()