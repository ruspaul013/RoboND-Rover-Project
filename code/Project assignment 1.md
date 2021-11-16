# Project assignment 1 Driving the rover in the simulator

## Your Challenges
Your first task is to add to the notebook **`Rover_Project_Test_Notebook.ipynb`** the perception steps and make a video with the data you have recorded in the manual mode. Then you should write the decision steps in **`decision.py`** and fill in **`perception.py`** and **`drive_rover.py`** so that the rover can map at least 40% of the environment and find at least one rock in the in the autonomous mode.

[NOTE] You need to write additional code to find the rock in the environment, that follow the same steps as in the case of the mapping:
- Apply a color threshold
- Apply the perspective transform
- Transform to rover centric coordinates
- Transform to world coordinates
- Convert to polar coordinates

And at the end:
- Mark the rock on the map using a white pixel

[BONUS] Find all the rock samples and pick them up.


## Solution
**`perception.py`**
- The first step was to define source and destination points for perspective transformation and then apply it on the images.
```
source = np.float32([[ 10, 135], [ 300,135 ], [ 200,94 ], [ 120, 94]])
destination = np.float32([[width/2 - square_size/2, height - square_size/2],
                                  [width/2 + square_size/2, height - square_size/2],
                                  [width/2 + square_size/2, height - square_size-bottom_offset], 
                                  [width/2 - square_size/2, height - square_size-bottom_offset]])
warped = perspect_transform(image, source, destination)
```

- In the laboratory, we learned to apply thresholds to detect the navigable terrain. In the images can appear obstacle or rocks. In this case we need to create 2 functions to detect them. We apply all the thresholds on the perspective image.
```
def rock_thresh(img, rgb_thresh=(100, 100, 70)):
    rock_select = np.zeros_like(img[:,:,0])
    rock_thresh_1 = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    rock_select[rock_thresh_1] = 1
    return rock_select
def obstacle_thresh(img, rgb_thresh=(140, 120, 150)):
    obstacle_select = np.zeros_like(img[:,:,0])
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    obstacle_select[below_thresh] = 1
    return obstacle_select
    
navigable_terrain = color_thresh(warped)
obstacle = obstacle_thresh(warped)
rock = rock_thresh(warped)
```
- Update the Robot vision. On every image that we applied thresholds we will have a certain plane of color (red for obstacles, green for rocks and blue for navigable terrain). The binary image needs to by multiplied with 255. The robots vision will be available in real time. 
```
Rover.vision_image[:,:,0] = obstacle * 255 #RED
Rover.vision_image[:,:,2] = navigable_terrain * 255 #BLUE
Rover.vision_image[:,:,1] = rock * 255 #GREEN
```
- Convert map image pixel values to rover-centric coords
```
x_rover,y_rover=rover_coords(navigable_terrain)
x_obs,y_obs=rover_coords(obstacle)
x_rock,y_rock = rover_coords(rock)
```
- Convert rover-centric pixel values to world coordinates
```
world_size=Rover.worldmap.shape[0]
scale=10
x_world, y_world = pix_to_world(x_rover, y_rover, Rover.pos[0], 
                                Rover.pos[1], Rover.yaw, 
                                world_size, scale)
x_obs_world, y_obs_world = pix_to_world(x_obs, y_obs, Rover.pos[0], 
                                Rover.pos[1], Rover.yaw, 
                                world_size, scale)
x_rock_world, y_rock_world= pix_to_world(x_rock, y_rock, Rover.pos[0], 
                                Rover.pos[1], Rover.yaw, 
                                world_size, scale)
```
- Update Rover worldmap (to be displayed on right side of screen). Rocks will appear as a white dot on the map, the navigable terrain will be blue and the obstacles will be red. 
```
Rover.worldmap[y_obs_world,x_obs_world,0] += 1
Rover.worldmap[y_world,x_world,2] += 1
Rover.worldmap[y_rock_world,x_rock_world,1] += 1
```
- Convert rover-centric pixel positions to polar coordinates
```
Rover.nav_dists, Rover.nav_angles = to_polar_coords(x_rover,y_rover)
```
**`decision.py`**
In the forward mode, a step was added in case that the robot gets stuck in an obstacle, there will be a waiting period of 3 seconds (90 frames). If the stuck time variable is greater than the default value, we choose to go backwards with the robot. We know that the robot it is stuck, when the velocity of the robot is smaller than 0.075.

We set a back_time frame counter to know how much time the robot needed to go backwards. When the value is over the limit, we stop the robot, set the throttle to positive value and reset the counters.
```
if ( Rover.vel <= 0.075 ):
        Rover.stuck_time += 1
    else:
        Rover.stuck_time = 0
            if (Rover.stuck_time > Rover.limit):
                Rover.throttle = -Rover.throttle_set
                Rover.steer = -np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                if (Rover.vel < 0):
                    Rover.back_time += 1
                    if (Rover.back_time > Rover.limit):
                        Rover.vel = 0
                        Rover.throttle = Rover.throttle_set
                        Rover.stuck_time = 0
                        Rover.back_time = 0
```
When the robot is near a rock, the robot will detect the sample, then stop and set the pick_up value to true. The robot will be able to collect the sample only if it will be in front of him when moving, in the rest of the cases, the rocks will be shown on the map with a white dot.
```
if Rover.near_sample and not Rover.picking_up:
    Rover.vel=0;
    Rover.brake = Rover.brake_set
    Rover.send_pickup = True
```
A solution to be able to collect more rocks will be to make the robot go to the location of the rocks, when they will appear in the robot vision (the rocks will be shown with green color)

**`drive_rover.py`**

Added 3 new variable, stuck_time ( a variable for frame counter when the robot is stuck because of an object), limit (default value for a frame counter) and back_time (a variable for a frame counter when the robot goes backwards)

**`Rover_Project_Test_Notebook.ypynb`**

We add the code from `perspective.py` to `Rover_Project_Test_Notebook.ypynb` in order to make a video and to update the worldmap using the csv file from the recorded sesion.

**Solution (kind of) proprosed by: Paul Rus**

*TO DO LIST:*
- solution to be able to collect more rocks
- in some cases, the robot will circle arround because of the perspective we set, therefore we need to find a solution in order to avoid this situation
- optimize the added code