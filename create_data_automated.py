import os
import magnum as mn
import numpy as np
import random
import json
import pickle
from pathlib import Path

# function to display the topdown map
from PIL import Image

import habitat_sim

import attr
import magnum as mn
import numpy as np
import quaternion  # noqa: F401

from habitat_sim import registry
from habitat_sim.agent import SceneNodeControl

from create_data import DataRecorder


class AutomatedDataRecorder(DataRecorder):
    def __init__(self, scene_id: int = 0, rec_type = "loiter"):
        super().__init__(scene_id=scene_id, rec_type = rec_type)

        self.rec_type = rec_type
        self.save = True

        
    def turn_degrees(self, deg: int, direction: str):
        """
        Turn the agent deg degrees to the left in descrete steps of <self.rotation_step> degrees.
        deg needs to be a multiple of <self.rotation_step> for full turn.
        
        directions: left, right
        """
        while deg > 0:
            self.observations = self.sim.step(f"turn_{direction}")
            deg -= self.rotation_step
            self.count += 1
            
            if self.save:
                self.save_images()
                self.save_count += 1
    
    def move_direction_meters(self, length: float, direction: str):
        """
        Move the agent <length> meters in descrete steps of <self.translation_step> meters
        in direction <direction>.

        directions: forward, backward, left, right, up, down
        """

        while length > 0:
            self.observations = self.sim.step(f"move_{direction}")
            length -= self.translation_step
            self.count += 1
            
            if self.save:
                self.save_images()
                self.save_count += 1

    
    def loiter_by_number_of_calls(self, num_calls: int):

        while num_calls > 0:
            self.observations = self.sim.step("loiter_right")
            num_calls -= 1
            self.count += 1
            
            if self.save:
                self.save_images()
                self.save_count += 1

    
    def look_up_down(self, deg: int, direction: str):
        """
        Look up or down by deg degrees in descrete steps of <self.rotation_step> degrees.
        deg needs to be a multiple of <self.rotation_step> for full turn.
        
        directions: up, down
        """
        while deg > 0:
            self.observations = self.sim.step(f"look_{direction}")
            deg -= self.rotation_step
            self.count += 1
            
            if self.save:
                self.save_images()
                self.save_count += 1

    
    def move_forward_turn_around(self):
        """
        Move forward and look around in a circle.
        """
        self.look_up_down(30, "down")
        self.look_up_down(20, "up")
        self.move_direction_meters(3.5, "forward")

        self.turn_degrees(200, "left")
        self.look_up_down(20, "down")


    def loiter_around(self, dst: int = 240):
        """
        moves around object in circle while keeping object in the center of the image
        """
        self.look_up_down(20, "down")

        self.loiter_by_number_of_calls(int(dst / 2))

        self.look_up_down(10, "down")
        self.look_up_down(10, "up")



def main():

    rec_type = "loiter" # "loiter", "turn"
    for scene_id in range(18):
        rec = AutomatedDataRecorder(scene_id, rec_type = rec_type)

        rec.spawn_object_infront_of_agent_by_string("statuette", dst = 1.7)
        rec.sim.step_physics(1.0)
       
        if rec_type == "turn":
            rec.move_forward_turn_around()
        elif rec_type == "loiter":
            rec.loiter_around()




if __name__ == "__main__":
    main()