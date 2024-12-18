import pyrealsense2 as rs
import numpy as np
import cv2
import os
import tifffile as tiff


# Path where images will be saved
save_path = './images/'
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480
FRAME_RATE = 30
BASE_RGB_NAME = 'image_rgb_'
BASE_DEPTH_NAME = 'image_xyz_'


os.makedirs(save_path, exist_ok=True)


# Function to find the next available iterator for filenames
def find_next_index(folder_path, base_name):
    files = os.listdir(folder_path)
    indices = [int(file.split('_')[-1].split('.')[0]) for file in files if file.startswith(base_name) and file.endswith(('.png', '.tif'))]
    if indices:
        return max(indices) + 1
    return 1


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs.stream.depth, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.z16, FRAME_RATE)
config.enable_stream(rs.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.bgr8, FRAME_RATE)

profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    # Determine the next index for files

    index = find_next_index(save_path, BASE_RGB_NAME)  # base_depth_name would work the same

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Get intrinsic parameters
        profile = aligned_depth_frame.get_profile()
        intr = profile.as_video_stream_profile().get_intrinsics()

        # Compute the XYZ coordinates
        xyz_image = np.zeros((480, 640, 3), dtype=np.float32)
        for y in range(480):
            for x in range(640):
                depth = depth_image[y, x]
                if depth:
                    depth_point = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)
                    xyz_image[y, x] = depth_point

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Save images
        cv2.imwrite(os.path.join(save_path, f'{BASE_RGB_NAME}{index:02}.png'), gray_image)
        tiff.imwrite(os.path.join(save_path, f'{BASE_DEPTH_NAME}{index:02}.tif'), xyz_image)
        print(f'Images saved as {BASE_RGB_NAME}{index}.png and {BASE_DEPTH_NAME}{index}.tif in {save_path}')
        index += 1  # Increment the file index
        input('Press a key to take another image or Ctrl+C to stop')

finally:
    # Stop streaming
    pipeline.stop()
