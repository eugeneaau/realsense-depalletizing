import numpy as np
import time
import cv2
import pyrealsense2 as rs
import random
import math
import argparse

from threading import Thread
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from sort import *

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes, RotatedBoxes

from detectron2.data import MetadataCatalog

import torch, torchvision

# Resolution of camera streams
RESOLUTION_X = 1280
RESOLUTION_Y = 720

# Configuration for histogram for depth image
NUM_BINS = 500
MAX_RANGE = 10000

AXES_SIZE = 10



class VideoStreamer:
    """
    Video streamer that takes advantage of multi-threading, and continuously is reading frames.
    Frames are then ready to read when program requires.
    """
    def __init__(self, video_file=None):
        """
        When initialised, VideoStreamer object should be reading frames
        """
        self.setup_image_config(video_file)
        self.configure_streams()
        self.stopped = False

    def start(self):
        """
        Initialise thread, update method will run under thread
        """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """
        Constantly read frames until stop() method is introduced
        """
        while True:

            if self.stopped:
                return

            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            # Convert image to numpy array and initialise images
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())


    def stop(self):
        self.pipeline.stop()
        self.stopped = True

    def read(self):
        return (self.color_image, self.depth_image)

    def setup_image_config(self, video_file=None):
        """
        Setup config and video steams. If --file is specified as an argument, setup
        stream from file. The input of --file is a .bag file in the bag_files folder.
        .bag files can be created using d435_to_file in the tools folder.
        video_file is by default None, and thus will by default stream from the 
        device connected to the USB.
        """
        config = rs.config()

        if video_file is None:
            
            config.enable_stream(rs.stream.depth, RESOLUTION_X, RESOLUTION_Y, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, RESOLUTION_X, RESOLUTION_Y, rs.format.bgr8, 30)
        else:
            try:
                config.enable_device_from_file("bag_files/{}".format(video_file))
            except:
                print("Cannot enable device from: '{}'".format(video_file))

        self.config = config

    def configure_streams(self):
        # Configure video streams
        self.pipeline = rs.pipeline()
    
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def get_depth_scale(self):
        return self.profile.get_device().first_depth_sensor().get_depth_scale()



class Predictor(DefaultPredictor):
    def __init__(self):
        self.config = self.setup_predictor_config()
        super().__init__(self.config)

    def create_outputs(self, color_image):
        self.outputs = self(color_image)

    def setup_predictor_config(self):
        """
        Setup config and return predictor. See config/defaults.py for more options
        """
        cfg = get_cfg()

        cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        # Mask R-CNN ResNet101 FPN weights
        cfg.MODEL.WEIGHTS = "model_final_a3ec72.pkl"
        # This determines the resizing of the image. At 0, resizing is disabled.
        cfg.INPUT.MIN_SIZE_TEST = 0

        return cfg

    def format_results(self, class_names):
        """
        Format results so they can be used by overlay_instances function
        """
        predictions = self.outputs['instances']
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        labels = None 
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        masks = predictions.pred_masks.cpu().numpy()
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

        boxes_list = boxes.tensor.tolist()
        scores_list = scores.tolist()
        class_list = classes.tolist()

        for i in range(len(scores_list)):
            boxes_list[i].append(scores_list[i])
            boxes_list[i].append(class_list[i])
        

        boxes_list = np.array(boxes_list)

        return (masks, boxes, boxes_list, labels, scores_list, class_list)    



class OptimizedVisualizer(Visualizer):
    """
    Detectron2's altered Visualizer class which converts boxes tensor to cpu
    """
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
    
    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.cpu().numpy()
        else:
            return np.asarray(boxes)



class DetectedObject:
    """
    Each object corresponds to all objects detected during the instance segmentation
    phase. Associated trackers, distance, position and velocity are stored as attributes
    of the object.
    masks[i], boxes[i], labels[i], scores_list[i], class_list[i]
    """
    def __init__(self, mask, box, label, score, class_name):
        self.mask = mask
        self.box = box
        self.label = label
        self.score = score
        self.class_name = class_name

    def __str__(self):
        ret_str = "The pixel mask of {} represents a {} and is {}m away from the camera.\n".format(self.mask, self.class_name, self.distance)
        if hasattr(self, 'track'):
            if hasattr(self.track, 'speed'):
                if self.track.speed >= 0:
                    ret_str += "The {} is travelling {}m/s towards the camera\n".format(self.class_name, self.track.speed)
                else:
                    ret_str += "The {} is travelling {}m/s away from the camera\n".format(self.class_name, abs(self.track.speed))
            if hasattr(self.track, 'impact_time'):
                ret_str += "The {} will collide in {} seconds\n".format(self.class_name, self.track.impact_time)
            if hasattr(self.track, 'velocity'):
                ret_str += "The {} is located at {} and travelling at {}m/s\n".format(self.class_name, self.track.position, self.track.velocity)
        return ret_str

    def create_vector_arrow(self):
        """
        Creates direction arrow which will use Arrow3D object. Converts vector to a suitable size so that the direction is clear.
        NOTE: The magnitude of the velocity is not represented through this arrow. The arrow lengths are almost all identical
        """
        arrow_ratio = AXES_SIZE / max(abs(self.track.velocity_vector[0]), abs(self.track.velocity_vector[1]), abs(self.track.velocity_vector[2]))
        self.track.v_points = [x * arrow_ratio for x in self.track.velocity_vector]

    

class Arrow3D(FancyArrowPatch):
    """
    Arrow used to demonstrate direction of travel for each object
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



def find_mask_centre(mask, color_image):
    """
    Finding centre of mask using moments
    """
    moments = cv2.moments(np.float32(mask))

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    return cX, cY


def find_median_depth(mask_area, num_median, histg):
    """
    Iterate through all histogram bins and stop at the median value. This is the
    median depth of the mask.
    """
    
    median_counter = 0
    centre_depth = "0.00"
    for x in range(0, len(histg)):
        median_counter += histg[x][0]
        if median_counter >= num_median:
            # Half of histogram is iterated through,
            # Therefore this bin contains the median
            centre_depth = x / 50
            break 

    return float(centre_depth)

def debug_plots(color_image, depth_image, mask, histg, depth_colormap):
    """
    This function is used for debugging purposes. This plots the depth color-
    map, mask, mask and depth color-map bitwise_and, and histogram distrobutions
    of the full image and the masked image.
    """
    full_hist = cv2.calcHist([depth_image], [0], None, [NUM_BINS], [0, MAX_RANGE])
    masked_depth_image = cv2.bitwise_and(depth_colormap, depth_colormap, mask= mask)

    plt.figure()
            
    plt.subplot(2, 2, 1)
    plt.imshow(depth_colormap)

    plt.subplot(2, 2, 2)
    plt.imshow(masks[i].mask)

    plt.subplot(2, 2, 3).set_title(labels[i])
    plt.imshow(masked_depth_image)

    plt.subplot(2, 2, 4)
    plt.plot(full_hist)
    plt.plot(histg)
    plt.xlim([0, 600])
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='type --file=file-name.bag to stream using file instead of webcam')
    args = parser.parse_args()

    # Initialise Detectron2 predictor
    predictor = Predictor()

    # Initialise video streams from D435
    video_streamer = VideoStreamer()

    # Initialise Kalman filter tracker from modified Sort module
    mot_tracker = Sort()

    depth_scale = video_streamer.get_depth_scale()
    print("Depth Scale is: {:.4f}m".format(depth_scale))

    speed_time_start = time.time()

    video_streamer.start()
    time.sleep(1)

    while True:
        
        time_start = time.time()
        color_image, depth_image = video_streamer.read()
        detected_objects = []

        t1 = time.time()

        camera_time = t1 - time_start
        
        predictor.create_outputs(color_image)
        outputs = predictor.outputs

        t2 = time.time()
        model_time = t2 - t1
        print("Model took {:.2f} time".format(model_time))

        predictions = outputs['instances']
        

        if outputs['instances'].has('pred_masks'):
            num_masks = len(predictions.pred_masks)
        else:
            # Even if no masks are found, the trackers must still be updated
            tracked_objects = mot_tracker.update(boxes_list)
            continue
        
        detectron_time = time.time()

        # Create a new Visualizer object from Detectron2 
        v = OptimizedVisualizer(color_image[:, :, ::-1], MetadataCatalog.get(predictor.config.DATASETS.TRAIN[0]))
        
        masks, boxes, boxes_list, labels, scores_list, class_list = predictor.format_results(v.metadata.get("thing_classes"))

        for i in range(num_masks):
            try:
                detected_obj = DetectedObject(masks[i], boxes[i], labels[i], scores_list[i], class_list[i])
            except:
                print("Object doesn't meet all parameters")
            
            detected_objects.append(detected_obj)


        tracked_objects = mot_tracker.update(boxes_list)

        
        v.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=None,
            assigned_colors=None,
            alpha=0.3
        )
        
        speed_time_end = time.time()
        total_speed_time = speed_time_end - speed_time_start
        speed_time_start = time.time()
        for i in range(num_masks):
            """
            Converting depth image to a histogram with num bins of NUM_BINS 
            and depth range of (0 - MAX_RANGE millimeters)
            """
        
            mask_area = detected_objects[i].mask.area()
            num_median = math.floor(mask_area / 2)
            
            histg = cv2.calcHist([depth_image], [0], detected_objects[i].mask.mask, [NUM_BINS], [0, MAX_RANGE])
            
            
            # Uncomment this to use the debugging function
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #debug_plots(color_image, depth_image, masks[i].mask, histg, depth_colormap)
            
            centre_depth = find_median_depth(mask_area, num_median, histg)
            detected_objects[i].distance = centre_depth
            cX, cY = find_mask_centre(detected_objects[i].mask._mask, v.output)

            # track refers to the list which holds the index of the detected mask which matches the tracker
            track = mot_tracker.matched[np.where(mot_tracker.matched[:,0]==i)[0],1]
            
            if len(track) > 0:
                # Index of detected mask
                track = track[0]
                if i not in mot_tracker.unmatched:
                    try:
                        # If the tracker's distance has already been initialised - tracker has been detected previously
                        if hasattr(mot_tracker.trackers[track], 'distance'):
                            mot_tracker.trackers[track].set_speed(centre_depth, total_speed_time)

                            mot_tracker.trackers[track].set_impact_time(centre_depth)

                            if mot_tracker.trackers[track].impact_time != False and mot_tracker.trackers[track].impact_time >= 0:
                                v.draw_text("{:.2f} seconds to impact".format(mot_tracker.trackers[track].impact_time), (cX, cY + 60))
                        
                        if hasattr(mot_tracker.trackers[track], 'position'):
                            # New 3D coordinates for current frame
                            x1, y1, z1 = rs.rs2_deproject_pixel_to_point(
                            video_streamer.depth_intrin, [cX, cY], centre_depth
                        )
                            
                            # Update states for tracked object
                            mot_tracker.trackers[track].set_velocity_vector(x1, y1, z1)
                            mot_tracker.trackers[track].set_distance_3d(x1, y1, z1)
                            mot_tracker.trackers[track].set_velocity(total_speed_time)

                            detected_objects[i].track = mot_tracker.trackers[track]

                            v.draw_text("{:.2f}m/s".format(detected_objects[i].track.velocity), (cX, cY + 40))

                            #relative_x = (cX - 64) / RESOLUTION_X
                            #relative_y = (abs(RESOLUTION_Y - cY) - 36) / RESOLUTION_Y

                            
                            # Show velocity vector arrow if velocity >= 1 m/s
                            """
                            if detected_objects[i].track.velocity >= 1:
                                ax = v.output.fig.add_axes([relative_x, relative_y, 0.1, 0.1], projection='3d')
                                ax.set_xlim([-AXES_SIZE, AXES_SIZE])
                                ax.set_ylim([-AXES_SIZE, AXES_SIZE])
                                ax.set_zlim([-AXES_SIZE, AXES_SIZE])
                                
                                #print(v_points)
                                detected_objects[i].create_vector_arrow()
                                a = Arrow3D([0, detected_objects[i].track.v_points[0]], [0, detected_objects[i].track.v_points[1]], [0, detected_objects[i].track.v_points[2]], mutation_scale=10, lw=1, arrowstyle="-|>", color="w")
                                ax.add_artist(a)
                                #ax.axis("off")
                                ax.set_facecolor((1, 1, 1, 0))
                                v.output.fig.add_axes(ax)
                            """

                        position = rs.rs2_deproject_pixel_to_point(
                            video_streamer.depth_intrin, [cX, cY], centre_depth
                        )    
                            
                        mot_tracker.trackers[track].set_distance(centre_depth)
                        mot_tracker.trackers[track].set_position(position)

                        
                    except IndexError:
                        continue


            v.draw_circle((cX, cY), (0, 0, 0))
            v.draw_text("{:.2f}m".format(centre_depth), (cX, cY + 20))
            

        #for i in detected_objects:
            #print(i)

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #cv2.imshow('Segmented Image', color_image)
        cv2.imshow('Segmented Image', v.output.get_image()[:, :, ::-1])
        #cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time_end = time.time()
        total_time = time_end - time_start

        print("Time to process frame: {:.2f}".format(total_time))
        print("FPS: {:.2f}\n".format(1/total_time))
        
    video_streamer.stop()
    cv2.destroyAllWindows()
