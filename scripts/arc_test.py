"""Leader and follow cars taking images of each other
   
   - Car 2 is leader, car 1 is follower
   - Car 1 tracks a goal pose that is a fixed distance behind car 2

Notes
-----
    Positive steer turns right, negative turns left

Settings: 'settings_two_car.json'

"""

#import airsim_data_collection.common.setup_path
import airsim
import os
import numpy as np
import time


## -------------------------- MAIN ------------------------ ##
if __name__ == "__main__":

    # Connect to client
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)

    car_controls = airsim.CarControls()

    # brake the car
    car_controls.brake = 1
    car_controls.throttle = 0
    client.setCarControls(car_controls)
    # wait until car is stopped
    time.sleep(1)

    # Arc parameters
    throttle = 0.5    
    dt = 0.1            # seconds
    arc_duration = 3.0  # seconds
    N = int(arc_duration / dt)
    N_arcs = 10
    steer_rates = np.linspace(-0.5, 0.5, N_arcs)
    arcs = np.zeros((N_arcs, N, 2))

    car_controls.brake = 0
    for i, steer_rate in enumerate(steer_rates):
        
        car_controls.steering = steer_rate 
        car_controls.throttle = throttle
        client.setCarControls(car_controls)

        for k in range(N):
            car_state = client.getCarState()
            arcs[i, k, 0] = car_state.kinematics_estimated.position.x_val
            arcs[i, k, 1] = car_state.kinematics_estimated.position.y_val
            time.sleep(dt)

    print(arcs)
    # try:
    #     print("driving routes")
    #     while(True):
    #         car_controls.steering = 0.5  #0.2 * np.sin(0.01 * count)
    #         car_controls.throttle = 1.0
    #         client.setCarControls(car_controls)


    # except KeyboardInterrupt:
    #     # Restore to original state
    #     client.reset()
    #     client.enableApiControl(False)


    # Restore to original state
    client.reset()
    client.enableApiControl(False)

    



