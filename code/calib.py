from camera import load_camera_calibration, save_camera_calibration, take_calibration_pictures
from ctu_crs import CRS97

robot = CRS97()

robot.initialize(home=True)

take_calibration_pictures(robot, divisions=(3,5,5))

calib_results = save_camera_calibration()

robot.close()
    
