import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

def main():
    # 1. ตั้งค่าการเชื่อมต่อกล้อง RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # เริ่มต้นการสตรีม
    profile = pipeline.start(config)
    
    # 2. กำหนดชุด Post-Processing Filters (นำ Decimation ออกเพื่อรักษา Resolution)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter() 
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    # ตั้งค่าการ Align (ซ้อนภาพ Depth ให้ตรงกับมุมมองของกล้อง Color)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 3. เตรียมหน้าต่างแสดงผล Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RealSense Point Cloud - Post-Processing Active", width=800, height=600)
    
    pointcloud = o3d.geometry.PointCloud()
    is_first_frame = True

    try:
        while True:
            # 4. รอรับเฟรมใหม่จากกล้อง
            frames = pipeline.wait_for_frames()
            
            # --- [แก้ไข] ทำการ Align ทันทีที่ได้เฟรมมา ---
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            # --- [แก้ไข] นำ Depth ที่ Align แล้ว มาผ่าน Filter ---
            filtered_depth = depth_to_disparity.process(aligned_depth_frame)
            filtered_depth = spatial.process(filtered_depth)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = disparity_to_depth.process(filtered_depth)
            filtered_depth = hole_filling.process(filtered_depth) # อุดรอยรั่วขั้นตอนสุดท้าย
            
            # แปลงข้อมูลเฟรมเป็น Numpy Array
            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # ดึงค่า Intrinsics จากเฟรม Depth ที่ถูกประมวลผลแล้ว
            depth_profile = filtered_depth.get_profile().as_video_stream_profile()
            intrinsics = depth_profile.get_intrinsics()
            
            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width, intrinsics.height, 
                intrinsics.fx, intrinsics.fy, 
                intrinsics.ppx, intrinsics.ppy
            )

            # 5. สร้าง Open3D RGBD Image
            o3d_color = o3d.geometry.Image(color_image_rgb)
            o3d_depth = o3d.geometry.Image(depth_image)
            
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, 
                depth_scale=1000.0, 
                depth_trunc=3.0, 
                convert_rgb_to_intensity=False
            )

            # 6. สร้าง/อัปเดต Point Cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, pinhole_camera_intrinsic
            )
            
            # กลับหัว Point Cloud ให้ตั้งตรงตามแกนปกติ
            pcd.transform([[1, 0, 0, 0], 
                           [0, -1, 0, 0], 
                           [0, 0, -1, 0], 
                           [0, 0, 0, 1]])

            # อัปเดตข้อมูลจุดลงใน Visualizer
            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors
            
            if is_first_frame:
                vis.add_geometry(pointcloud)
                is_first_frame = False
            else:
                vis.update_geometry(pointcloud)
                
            # อัปเดตหน้าต่าง UI
            vis.poll_events()
            vis.update_renderer()

    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()