#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.transform import rescale
from skimage import img_as_float32, img_as_ubyte, img_as_float64
from skimage.transform import hough_circle
from skimage.morphology import dilation, erosion
from skimage.morphology import  remove_small_objects, remove_small_holes
from skimage.exposure import  equalize_hist
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import cv2 as cv
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage import img_as_float32, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import dilation, erosion, area_closing, area_opening
from skimage.morphology import disk, square, rectangle, remove_small_objects, remove_small_holes
from skimage.morphology import reconstruction
from skimage.exposure import equalize_adapthist, equalize_hist


# In[156]:


class Walker:
    
    def __init__(self, init_pos, main_angle, step_size, allowed_pixels_mask, num_of_directions=3):
        self.pos = init_pos
        self.step_size = step_size
        self.allowed_pixels_mask = np.array(copy.deepcopy(allowed_pixels_mask), dtype=np.bool)
        self.num_of_directions = num_of_directions
        self.directions = self.parse_main_angle(main_angle)
        self.accumulated_distance = 0
            
    def parse_main_angle(self, main_angle):
        directions = {
            # angle: (y, x)
            0: (0, self.step_size),
            45: (-self.step_size, self.step_size),
            90: (-self.step_size, 0),
            135: (-self.step_size, -self.step_size),
            180: (0, -self.step_size),
            225: (self.step_size, -self.step_size),
            270: (self.step_size, 0),
            315: (self.step_size, self.step_size),
            360: (0, self.step_size),
        }
        
        closest_angles = sorted(
            list(directions.keys()), 
            key=lambda angle: abs(angle - main_angle), reverse=True
        )[:self.num_of_directions + 1]
        
        if 0 in closest_angles and 360 in closest_angles:
            closest_angles.remove(360)
        
        directions = {k: v for k, v in directions.items() if k in closest_angles}
            
        return directions
        
    def get_feasible_directions(self):
        feasible_directions = dict()
        
        for angle, (step_y, step_x) in self.directions.items():
            new_y = int(self.pos[0] + step_y)
            new_x = int(self.pos[1] + step_x)
            
            if 0 <= new_y < self.allowed_pixels_mask.shape[0]                 and 0 <= new_x < self.allowed_pixels_mask.shape[1]                 and self.allowed_pixels_mask[new_y, new_x]:
                
                feasible_directions[angle] = (step_y, step_x)
                
        return feasible_directions if len(feasible_directions) > 0 else None
        
    def step(self):
        fdirs = self.get_feasible_directions()
        if fdirs is None:
            return None  # Walker dies here

        rkey = np.random.choice(list(fdirs.keys()))
        self.pos[0] += fdirs[rkey][0]
        self.pos[1] += fdirs[rkey][1]
        
        self.accumulated_distance += np.linalg.norm(np.array(fdirs[rkey]))
        
        return self.pos


# In[157]:


class WalkersRoy:
    
    def __init__(
        self
        , city_pos
        , other_city_centers
        , old_city_centers
        , allowed_pixels_mask
        , walkers_directions_count=10
        , walkers_density_per_direction=5
        , walkers_step_size=10
        , threshold_cities_diff=30
        , num_of_directions=3
    ):
        
        self.neigbouring_cities_distances = defaultdict(list)
        main_directions = np.linspace(0, 360, walkers_directions_count)
        
        walkers = []
        for md in main_directions:
            walkers.append([
                Walker(
                    copy.deepcopy(city_pos)
                    , md
                    , walkers_step_size
                    , copy.deepcopy(allowed_pixels_mask)
                    , num_of_directions
                ) 
                for _ in range(walkers_density_per_direction)
            ])
            
        self.other_city_centers = other_city_centers
        self.old_city_centers = old_city_centers
        self.walkers_step_size = walkers_step_size
        self.walkers = walkers
        self.threshold_cities_diff = threshold_cities_diff
        
    def step(self):
        for i, _ in enumerate(self.walkers):
            for j, _ in enumerate(self.walkers[i]):
                pos = self.walkers[i][j].step()
                
                if pos is None:
                    self.walkers[i][j] = None
                else:
                    for oc in self.other_city_centers:
                        d = np.linalg.norm(pos - oc)
                        if d < self.threshold_cities_diff:
                            self.neigbouring_cities_distances[f'{oc[0]}_{oc[1]}'].append(
                                self.walkers[i][j].accumulated_distance
                            )
                            
                            self.walkers[i][j] = None
                            
                    for oc in self.old_city_centers:
                        d = np.linalg.norm(pos - oc)
                        if d < self.threshold_cities_diff:
                            # NOTE: That's rude I know
                            self.walkers[i][j] = None
        
        for i, _ in enumerate(self.walkers):
            self.walkers[i] = [a for a in self.walkers[i] if a is not None]
                        
        overall_alive_walkers_count = 0
        for i, _ in enumerate(self.walkers):
            overall_alive_walkers_count += len(self.walkers[i])
            
        if overall_alive_walkers_count == 0:
            return 'finished'
        else:
            return 'in progress'


# In[170]:


def detect_centers(img):
    img_gr = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    circs = cv.HoughCircles(
        img_gr, cv.HOUGH_GRADIENT, 1, 100 , param1=50, param2=32, minRadius=22, maxRadius=31
    )

    centers = None
    if circs is not None:
        circ = np.uint16(np.around(circs[0]))
        centers = np.array(circ[:, :-1].tolist())
        centers = centers[:, [1,0]]

    return centers

# def scoring(mask):
#     contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     areas = [cv.contourArea(cnt) for cnt in contours]
#     avg_area = min(np.median(areas), 275)
#     high_bound = avg_area + 75
#     low_bound = avg_area - 75
#     scor = 0
#     for area in areas:
#         if area >= low_bound and area <= high_bound: 
#             scor+=1
#         elif area > high_bound: 
#             res= area / avg_area
#             if(res>1) & (res<=2.2):
#                 scor+=2
#             elif(res>2) & (res<=3.2): #3 bricks
#                 scor+=4
#             elif (res>3.2) & (res<5.5):
#                 scor+=7
#             elif (res>=5.5) & (res<=7):
#                 scor+=15
#             elif res>=8:
#                 scor+=21
#             cof = (area / avg_area - area // avg_area)
#             if cof >= 0.5:
#                 scor += 1
#     return scor
def _calculate_score(neigbouring_cities_distances, block_length=50):
    
    bricks2scores = {
        0: 0,
        1: 1,
        2: 2,
        3: 4,
        4: 7,
        6: 15,
        8: 21
    }
    brick_path_lens = np.array(list(bricks2scores.keys()))
    
    for key in neigbouring_cities_distances:
        neigbouring_cities_distances[key] = np.array(neigbouring_cities_distances[key]).mean()   
    bricks_counts = [np.round(v / block_length) for v in neigbouring_cities_distances.values()]
    score = 0
    for bricks_count in bricks_counts:
        i = np.argsort(np.abs(brick_path_lens - bricks_count))[0]
        score += bricks2scores[brick_path_lens[i]]
    
    return int(score)

def calculate_scores(
    centers
    , bricks_map
    , walkers_directions_count=25
    , walkers_density_per_direction=15
    , walkers_step_size=10
    , threshold_cities_diff=31
    , num_of_directions=3
    , max_iter=250
    , city_radius=18
    , centers_multiplier=0.25
    , dilation_radius=5
    , brick_length=41
):
    
    # Scaling centers
    centers = (centers * centers_multiplier).astype(int)
    
    # Draw cities circles and dilate map to make it smooth for walkers
    bricks_map = dilation(bricks_map, disk(dilation_radius))
    for center in centers:
        city_disk = img_as_ubyte(disk(city_radius, dtype=float))
        bricks_patch = bricks_map[
            int(center[0]) - city_radius : int(center[0]) + city_radius + 1
            , int(center[1]) - city_radius : int(center[1]) + city_radius + 1
        ]
        bricks_map[
            int(center[0]) - city_radius : int(center[0]) + city_radius + 1
            , int(center[1]) - city_radius : int(center[1]) + city_radius + 1
        ] = np.clip(city_disk + bricks_patch, 0, 255)
        
    # Counting scores for each city
    accumulated_score = 0
    for i, center in centers[:-1]:
        other_centers = centers[i + 1:]
        old_centers = centers[:i]
                
        roy = WalkersRoy(
            copy.deepcopy(center)
            , copy.deepcopy(other_centers)
            , copy.deepcopy(old_centers)
            , allowed_pixels_mask = copy.deepcopy(bricks_map)
            , walkers_directions_count = walkers_directions_count
            , walkers_density_per_direction = walkers_density_per_direction
            , walkers_step_size = walkers_step_size
            , threshold_cities_diff = threshold_cities_diff
            , num_of_directions = num_of_directions
        )

        for j in range(max_iter):
            status = roy.step()
            if status == 'finished':
                break
                
        score = _calculate_score(roy.neigbouring_cities_distances, block_length=brick_length)
        accumulated_score += score

    return int(accumulated_score)

    



            
def count_trains(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(cnt) for cnt in contours]
    
    avg_area = min(np.median(areas), 275)
    high_bound = avg_area + 75
    low_bound = avg_area - 75

    count = 0
    for area in areas:
        if area >= low_bound and area <= high_bound: 
            count += 1
        elif area > high_bound: 
            count += area // avg_area
            cof = (area / avg_area - area // avg_area)
            if cof >= 0.5:
                count += 1
    
    return count


def detect_red_trains(img):
    img = img.copy()
    HLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    mask = ((HUE < 0.35) | (HUE > 350)) & (SAT > 0.7)  # & (LIGHT > 0.25)
    mask = erosion(mask)
    mask = remove_small_objects(mask, 128)
    mask = dilation(mask)
    mask = img_as_ubyte(mask.astype(np.float32))

    return mask


def detect_blue_trains(img):

    img = img.copy()
    HLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    SAT = HLS[:, :, 2]

    mask = ((HUE < 216.8) & (HUE > 205)) & (SAT > 0.7)
    mask = erosion(mask)
    mask = remove_small_objects(mask, 128)
    mask = dilation(mask)
    mask = img_as_ubyte(mask.astype(np.float32))

    return mask


def detect_green_trains(img):
    img = img.copy()
    HLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    SAT = HLS[:, :, 2]

    mask = ((HUE < 160) & (HUE > 140)) 
    mask = erosion(mask)
    mask = remove_small_objects(mask, 128)
    mask = dilation(mask)
    mask = img_as_ubyte(mask.astype(np.float32))

    return mask


def detect_black_trains(img):
    img = img.copy()
    HLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    mask = (LIGHT < 0.093) & (SAT < 0.5)
    mask = erosion(mask)
    mask = remove_small_objects(mask, 128)
    mask = dilation(mask)
    mask = img_as_ubyte(mask.astype(np.float32))

    return mask


def detect_yellow_trains(img):
    img = img.copy()
    HLS = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    SAT = HLS[:, :, 2]

    mask = ((HUE < 60) & (HUE > 35)) & (SAT > 0.82)
    mask = erosion(mask)
    mask = remove_small_objects(mask, 128)
    mask = dilation(mask)
    mask = img_as_ubyte(mask.astype(np.float32))

    return mask


def predict_image(img):
    #Process1: City Detection:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    centrs = detect_centers(img)
#Process2: Markers detection and counting using functions of detection and the count function
    img = rescale(img, (0.25, 0.25, 1))
    img = img_as_float32(equalize_hist(img))
    red = count_trains(detect_red_trains(img))
    green = count_trains(detect_green_trains(img))
    blue = count_trains(detect_blue_trains(img))
    yellow = count_trains(detect_yellow_trains(img))   
    black = count_trains(detect_black_trains(img))
    red_score=calculate_scores(centrs,detect_red_trains(img))
    green_score=calculate_scores(centrs,detect_green_trains(img))
    blue_score=calculate_scores(centrs,detect_blue_trains(img))
    yellow_score=calculate_scores(centrs,detect_yellow_trains(img))
    black_score=calculate_scores(centrs,detect_black_trains(img))
    
    if red < 5: red = 0
    if green < 5: green = 0
    if blue < 5: blue = 0
    if yellow < 5: yellow = 0
    if black < 5: black = 0
    if red_score < 5: red = 0
    if green_score < 5: green_score = 0
    if blue_score < 5: blue_score = 0
    if yellow_score < 5: yellow_score = 0
    if black_score < 5: black_score = 0

    players = {  'red': red , 'green': green, 'blue': blue  , 'yellow': yellow  , 'black': black
    }
    
    score = {"blue": blue_score, "green": green_score, "black":black_score, "yellow": yellow_score, "red": red_score}
        
    return centrs, players, score


# In[176]:


img = cv.imread('Desktop/Intro to CV/black_blue_green.jpg')


# In[177]:


a,b,c=predict_image(img)


# In[173]:


#a


# In[174]:


#b


# In[175]:


#c


# In[133]:





# In[ ]:




