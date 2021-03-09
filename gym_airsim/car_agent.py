from airsim import CarClient, CarControls, ImageRequest, ImageType
from configparser import ConfigParser
import numpy as np
import natsort
from numpy.linalg import norm
from os.path import dirname, abspath, join
import cv2
from datetime import datetime



class CarAgent(CarClient):
    def __init__(self):
        # connect to the AirSim simulator
        super().__init__()
        super().confirmConnection()
        super().enableApiControl(True)

        # read configuration
        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))
        
        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])
        self.image_size = self.image_height * self.image_width * self.image_channels
        self.throttle = float(config['car_agent']['fixed_throttle'])

        # fetch waypoints
        waypoint_regex = config['airsim_settings']['waypoint_regex']
        self._fetchWayPoints(waypoint_regex)
   
    def restart(self):
        super().reset()
        super().enableApiControl(True)
        
    # get RGB image from the front camera    
    def observe(self):
        size = 0
        # Sometimes simGetImages() return an empty list.  If so, try it again.
        while size != self.image_size:
            response = super().simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
            img_rgb_1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_1d.reshape(response.height, response.width, 3)
            #crop image to throw unnecessary data
            img_rgb = img_rgb[90:, :, :]
            size = img_rgb.size

            # save cropped images to see what the car sees in 'record' folder
            #cv2.imwrite('record/' + datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S.%f") + ".png", img_rgb)

        img3d_rgb = img_rgb.reshape(self.image_height, self.image_width, self.image_channels)
        return img3d_rgb

    def move(self, action):
        # rescaling action from actionspace of (-1) to (1) to (-0.5) to (0.5) to match car's settings in UE4
        car_controls = CarControls(throttle=self.throttle,steering=(float(action[0]/2)))
        super().setCarControls(car_controls)

    def _fetchWayPoints(self, waypoint_regex):
        # get all objects with names starts with WayPoint
        wp_names = super().simListSceneObjects(waypoint_regex)
        # sort waypoints according to name and index
        natsort.natsorted(wp_names)
        vec2r_to_numpy_array = lambda vec: np.array([vec.x_val, vec.y_val])
        self.waypoints = []
        for wp in wp_names:
            pose = super().simGetObjectPose(wp)
            self.waypoints.append(vec2r_to_numpy_array(pose.position))
        return
        
    def simGetWayPoints(self):
        return self.waypoints
    
    def simGet2ClosestWayPoints(self):
        total_distance = lambda p, p1, p2: norm(p-p1) + norm(p-p2)
        pos = super().simGetVehiclePose().position
        car_point = np.array([pos.x_val, pos.y_val])
        
        min_dist = 9999999 
        min_i = 0
        for i in range(len(self.waypoints)-1):
            dist = total_distance(car_point, self.waypoints[i], self.waypoints[i+1])
            if dist < min_dist:
                min_dist = dist
                min_i = i
        
        return self.waypoints[min_i], self.waypoints[min_i+1]