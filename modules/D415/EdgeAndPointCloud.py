import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# 1. ตั้งค่า Pipeline สำหรับกล้อง D415
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# เริ่มการเชื่อมต่อกล้อง
profile = pipeline.start(config)

# 2. ตั้งค่า Passive Stereo (ปิด IR Projector)
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    print("IR Projector Disabled: Running in Passive Stereo mode")

# สร้างออบเจ็กต์สำหรับ Align ภาพ Depth ให้ตรงกับ RGB
align_to = rs.stream.color
align = rs.align(align_to)

# 3. ดึงค่า Intrinsics ของกล้องสำหรับ Open3D
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
)

# 4. ตั้งค่าหน้าต่าง 3D (Open3D)
vis = o3d.visualization.Visualizer()
vis.create_window("3D Point Cloud (Passive Stereo - Depth Colored)", width=800, height=600)
pcd = o3d.geometry.PointCloud()
is_first_frame = True

try:
    print("กำลังสตรีมภาพ... กด 'q' ที่หน้าต่าง 2D เพื่อออก")
    while True:
        # รอรับเฟรมภาพและจับคู่
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # แปลงเป็น Numpy Array
        depth_data = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # ==========================================
        # ส่วนที่ 1: การประมวลผล 2D และจัดหน้าต่าง 2x2
        # ==========================================
        
        # 1.1 ภาพ Depth Colormap
        depth_normalized = cv2.normalize(depth_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap_2d = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 1.2 สกัดขอบ (Edges)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 1.3 สร้างภาพซ้อนทับ (Overlay) ให้เส้นขอบเป็นสีเขียวบนภาพต้นฉบับ
        overlay_image = color_image.copy()
        overlay_image[edges == 255] = [0, 255, 0] # กำหนดให้พิกเซลที่เป็นขอบกลายเป็นสีเขียว (BGR)

        # 1.4 นำภาพมาต่อกันแบบ 2x2 Grid
        # แถวบน: Original RGB กับ Depth
        top_row = np.hstack((color_image, depth_colormap_2d))
        # แถวล่าง: Edges (ขาว-ดำ) กับ Overlay (เส้นขอบสีเขียว)
        bottom_row = np.hstack((edges_3channel, overlay_image))
        
        # นำแถวบนและแถวล่างมาต่อกันในแนวตั้ง
        grid_2x2 = np.vstack((top_row, bottom_row))

        # ลดขนาดหน้าต่างลงครึ่งหนึ่ง เพื่อไม่ให้ล้นหน้าจอ (1280x960 -> 640x480)
        # หากต้องการดูความละเอียดเต็ม สามารถลบบรรทัด cv2.resize ออกได้
        grid_2x2_resized = cv2.resize(grid_2x2, (640, 480))

        cv2.imshow('2x2 Grid: RGB | Depth | Edges | Overlay', grid_2x2_resized)

        # ==========================================
        # ส่วนที่ 2: การสร้าง 3D Point Cloud ด้วย Open3D
        # ==========================================
        depth_colormap_rgb = cv2.cvtColor(depth_colormap_2d, cv2.COLOR_BGR2RGB)
        
        o3d_color = o3d.geometry.Image(depth_colormap_rgb)
        o3d_depth = o3d.geometry.Image(depth_data)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, convert_rgb_to_intensity=False)
        
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic)
        
        temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        pcd.points = temp_pcd.points
        pcd.colors = temp_pcd.colors

        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()

        # กด 'q' ที่หน้าต่าง OpenCV เพื่อหยุด
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()