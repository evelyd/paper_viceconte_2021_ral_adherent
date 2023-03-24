# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# Use tf version 2.3.0 as 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import math
import json
import yarp
import numpy as np
from typing import List, Dict
from scenario import gazebo as scenario
from dataclasses import dataclass, field
from gym_ignition.rbd.idyntree import numpy
from gym_ignition.rbd.conversions import Rotation
from gym_ignition.rbd.conversions import Transform
from adherent.MANN.utils import denormalize
from gym_ignition.rbd.conversions import Quaternion
from adherent.MANN.utils import read_from_file
from adherent.data_processing.utils import iCub
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.trajectory_generation.utils import trajectory_blending
from adherent.trajectory_generation.utils import load_output_mean_and_std
from adherent.trajectory_generation.utils import load_component_wise_input_mean_and_std

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt


@dataclass
class StorageHandler:
    """Class to store all the quantities relevant in the trajectory generation pipeline and save data."""

    # Storage paths for the footsteps, postural, joystick input and blending coefficients
    footsteps_path: str
    postural_path: str
    joystick_input_path: str
    blending_coefficients_path: str

    # Storage dictionaries for footsteps, postural, joystick input and blending coefficients
    footsteps: Dict = field(default_factory=lambda: {'l_foot': [], 'r_foot': []})
    posturals: Dict = field(default_factory=lambda: {'base': [], 'joints': [], 'links': [], 'com': []})
    joystick_inputs: Dict = field(default_factory=lambda: {'raw_data': [], 'base_velocities': [], 'base_angular_velocities': []})
    blending_coeffs: Dict = field(default_factory=lambda: {'w_1': [], 'w_2': [], 'w_3': [], 'w_4': []})

    @staticmethod
    def build(storage_path: str) -> "StorageHandler":
        """Build an instance of StorageHandler."""

        # Storage paths for the footsteps, postural, joystick input and blending coefficients
        footsteps_path = os.path.join(storage_path, "footsteps.txt")
        postural_path = os.path.join(storage_path, "postural.txt")
        joystick_input_path = os.path.join(storage_path, "joystick_input.txt")
        blending_coefficients_path = os.path.join(storage_path, "blending_coefficients.txt")

        return StorageHandler(footsteps_path,
                              postural_path,
                              joystick_input_path,
                              blending_coefficients_path)

    def update_joystick_inputs_storage(self, raw_data: List, base_velocities: List, base_angular_velocities: List) -> None:
        """Update the storage of the joystick inputs."""

        self.joystick_inputs["raw_data"].append(raw_data)
        self.joystick_inputs["base_velocities"].append(base_velocities)
        self.joystick_inputs["base_angular_velocities"].append(base_angular_velocities)

    def update_blending_coefficients_storage(self, blending_coefficients: List) -> None:
        """Update the storage of the blending coefficients."""

        self.blending_coeffs["w_1"].append(float(blending_coefficients[0][0]))
        self.blending_coeffs["w_2"].append(float(blending_coefficients[0][1]))
        self.blending_coeffs["w_3"].append(float(blending_coefficients[0][2]))
        self.blending_coeffs["w_4"].append(float(blending_coefficients[0][3]))

    def update_footsteps_storage(self, support_foot: str, footstep: Dict) -> None:
        """Add a footstep to the footsteps storage."""

        self.footsteps[support_foot].append(footstep)

    def replace_footsteps_storage(self, footsteps: Dict) -> None:
        """Replace the storage of footsteps with an updated footsteps list."""

        self.footsteps = footsteps

    def update_posturals_storage(self, base: Dict, joints: Dict, links: Dict, com: List) -> None:
        """Update the storage of the posturals."""

        self.posturals["base"].append(base)
        self.posturals["joints"].append(joints)
        self.posturals["links"].append(links)
        self.posturals["com"].append(com)

    def save_data_as_json(self) -> None:
        """Save all the stored data using the json format."""

        # Save footsteps
        with open(self.footsteps_path, 'w') as outfile:
            json.dump(self.footsteps, outfile)

        # Save postural
        with open(self.postural_path, 'w') as outfile:
            json.dump(self.posturals, outfile)

        # Save joystick inputs
        with open(self.joystick_input_path, 'w') as outfile:
            json.dump(self.joystick_inputs, outfile)

        # Save blending coefficients
        with open(self.blending_coefficients_path, 'w') as outfile:
            json.dump(self.blending_coeffs, outfile)

        # Debug
        input("\nData have been saved. Press Enter to continue the trajectory generation.")


@dataclass
class FootstepsExtractor:
    """Class to extract the footsteps from the generated trajectory."""

    # Auxiliary variables for the footsteps update before saving
    nominal_DS_duration: float
    difference_position_threshold: float

    # Auxiliary variables to handle the footsteps deactivation time
    difference_height_norm_threshold: bool
    waiting_for_deactivation_time: bool = False

    @staticmethod
    def build(nominal_DS_duration: float = 0.04,
              difference_position_threshold: float = 0.04,
              difference_height_norm_threshold: bool = 0.005) -> "FootstepsExtractor":
        """Build an instance of FootstepsExtractor."""

        return FootstepsExtractor(nominal_DS_duration=nominal_DS_duration,
                                  difference_position_threshold=difference_position_threshold,
                                  difference_height_norm_threshold=difference_height_norm_threshold)

    def should_update_footstep_deactivation_time(self, kindyn: kindyncomputations.KinDynComputations) -> bool:
        """Check whether the deactivation time of the last footstep needs to be updated."""

        # Retrieve the transformation from the world frame to the base frame
        world_H_base = kindyn.get_world_base_transform()

        # Compute right foot height
        base_H_r_foot = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="r_foot")
        W_H_RF = world_H_base.dot(base_H_r_foot)
        W_right_foot_pos = W_H_RF [0:3, -1]
        right_foot_height = W_right_foot_pos[2]

        # Compute left foot height
        base_H_l_foot = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="l_foot")
        W_H_LF = world_H_base.dot(base_H_l_foot)
        W_left_foot_pos = W_H_LF[0:3, -1]
        left_foot_height = W_left_foot_pos[2]

        # Compute the difference in height between the feet
        difference_height_norm = np.linalg.norm(left_foot_height - right_foot_height)

        # If the height difference is above a threshold and a foot is being detached, the deactivation
        # time of the last footstep related to the detaching foot needs to be updated
        if self.waiting_for_deactivation_time and difference_height_norm > self.difference_height_norm_threshold:
            self.waiting_for_deactivation_time = False
            return True

        return False

    def create_new_footstep(self, kindyn: kindyncomputations.KinDynComputations,
                            support_foot: str, activation_time: float) -> Dict:
        """Retrieve the information related to a new footstep."""

        new_footstep = {}

        # Compute new footstep 3D and 2D position
        world_H_base = kindyn.get_world_base_transform()
        base_H_support_foot = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=support_foot)
        W_H_SF = world_H_base.dot(base_H_support_foot)
        support_foot_pos = W_H_SF[0:3, -1]
        support_foot_ground_pos = [support_foot_pos[0], support_foot_pos[1]]
        new_footstep["3D_pos"] = list(support_foot_pos)
        new_footstep["2D_pos"] = support_foot_ground_pos

        # Compute new footstep 3D and 2D orientation
        support_foot_quat = Quaternion.from_matrix(W_H_SF[0:3, 0:3])
        W_R_SF = Rotation.from_quat(Quaternion.to_xyzw(np.asarray(support_foot_quat)))
        W_RPY_SF = Rotation.as_euler(W_R_SF, 'xyz')
        new_footstep["3D_orient"] = W_R_SF.as_matrix().tolist()
        new_footstep["2D_orient"] = W_RPY_SF[2]

        # Assign new footstep activation time
        new_footstep["activation_time"] = activation_time

        # Use a temporary flag indicating that the deactivation time has not been computed yet
        new_footstep["deactivation_time"] = -1

        # Set the flag indicating that the last footstep has no deactivation time yet accordingly
        self.waiting_for_deactivation_time = True

        return new_footstep

    def update_footsteps(self, final_deactivation_time: float, footsteps: Dict) -> Dict:
        """Update the footsteps list before saving data by replacing temporary deactivation times (if any) and
        merging footsteps which are too close each other in order to avoid unintended footsteps on the spot.
        """

        # Update the deactivation time of the last footstep of each foot (they need to coincide to be processed
        # properly in the trajectory control layer)
        for foot in footsteps.keys():
            footsteps[foot][-1]["deactivation_time"] = final_deactivation_time

        # Replace temporary deactivation times in the footsteps list (if any)
        updated_footsteps = self.replace_temporary_deactivation_times(footsteps=footsteps)

        # Merge footsteps which are too close each other
        updated_footsteps = self.merge_close_footsteps(final_deactivation_time=final_deactivation_time,
                                                       footsteps=updated_footsteps)

        return updated_footsteps

    def replace_temporary_deactivation_times(self, footsteps: Dict) -> Dict:
        """Replace temporary footstep deactivation times that may not have been updated properly."""

        # Map from one foot to the other
        other_foot = {"l_foot": "r_foot", "r_foot": "l_foot"}

        for foot in ["l_foot","r_foot"]:

            for footstep in footsteps[foot]:

                # If a temporary footstep deactivation time is detected
                if footstep["deactivation_time"] == -1:

                    # Retrieve the footstep activation time
                    current_activation_time = footstep["activation_time"]

                    for footstep_other_foot in footsteps[other_foot[foot]]:

                        # Retrieve the activation time of the next footstep of the other foot
                        other_foot_activation_time = footstep_other_foot["activation_time"]

                        if other_foot_activation_time > current_activation_time:

                            # Update the deactivation time so to have a double support (DS) phase of the nominal duration
                            current_deactivation_time = other_foot_activation_time + self.nominal_DS_duration
                            footstep["deactivation_time"] = current_deactivation_time

                            break

        return footsteps

    def merge_close_footsteps(self, final_deactivation_time: float, footsteps: Dict) -> Dict:
        """Merge footsteps that are too close each other in order to avoid unintended footsteps on the spot."""

        # Initialize updated footsteps list
        updated_footsteps = {"l_foot": [], "r_foot": []}

        for foot in footsteps.keys():

            # Auxiliary variable to handle footsteps update
            skip_next_contact = False

            for i in range(len(footsteps[foot]) - 1):

                if skip_next_contact:
                    skip_next_contact = False
                    continue

                # Compute the norm of the difference in position between consecutive footsteps of the same foot
                current_footstep_position = np.array(footsteps[foot][i]["2D_pos"])
                next_footstep_position = np.array(footsteps[foot][i + 1]["2D_pos"])
                difference_position = np.linalg.norm(current_footstep_position - next_footstep_position)

                if difference_position >= self.difference_position_threshold:

                    # Do not update footsteps which are not enough close each other
                    updated_footsteps[foot].append(footsteps[foot][i])

                else:

                    # Merge footsteps which are close each other: the duration of the current footstep is extended
                    # till the end of the subsequent footstep
                    updated_footstep = footsteps[foot][i]
                    updated_footstep["deactivation_time"] = footsteps[foot][i + 1]["deactivation_time"]
                    updated_footsteps[foot].append(updated_footstep)
                    skip_next_contact = True

            # If the last updated footstep ends before the final deactivation time, add the last original footstep
            # to the updated list of footsteps
            if updated_footsteps[foot][-1]["deactivation_time"] != final_deactivation_time:
                updated_footsteps[foot].append(footsteps[foot][-1])

        return updated_footsteps


@dataclass
class PosturalExtractor:
    """Class to extract the postural from the generated trajectory."""

    @staticmethod
    def build() -> "PosturalExtractor":
        """Build an instance of PosturalExtractor."""

        return PosturalExtractor()

    @staticmethod
    def create_new_posturals(base_position: List, base_quaternion: List, joint_positions: List, controlled_joints: List,
                             kindyn: kindyncomputations.KinDynComputations, link_names: List) -> (List, List, List, List):
        """Retrieve the information related to a new set of postural terms."""

        # Store the postural term related to the base position and orientation
        new_base_postural = {"postion": list(base_position), "wxyz_quaternions": list(base_quaternion)}

        # Store the postural term related to the joint angles
        new_joints_postural = {controlled_joints[k]: joint_positions[k] for k in range(len(controlled_joints))}

        # Store the postural term related to the link orientations
        new_links_postural = {}
        world_H_base = kindyn.get_world_base_transform()
        for link_name in link_names:
            base_H_link = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=link_name)
            world_H_link = world_H_base.dot(base_H_link)
            new_links_postural[link_name] = list(Quaternion.from_matrix(world_H_link[0:3, 0:3]))

        # Store the postural term related to the com positions
        new_com_postural = list(kindyn.get_com_position())

        return new_base_postural, new_joints_postural, new_links_postural, new_com_postural


@dataclass
class KinematicComputations:
    """Class for the kinematic computations exploited within the trajectory generation pipeline to compute
    kinematically-feasible base motions.
    """

    kindyn: kindyncomputations.KinDynComputations

    # Footsteps and postural extractors
    footsteps_extractor: FootstepsExtractor
    postural_extractor: PosturalExtractor

    # Simulated robot (for visualization only)
    icub: iCub
    controlled_joints: List
    gazebo: scenario.GazeboSimulator

    # Support foot and support vertex related quantities
    local_foot_vertices_pos: List
    support_vertex_prev: int = 0
    support_vertex: int = 0
    support_foot_prev: str = "r_foot"
    support_foot: str = "r_foot"
    support_foot_pos: float = 0
    support_vertex_pos: float = 0
    support_vertex_offset: float = 0

    @staticmethod
    def build(kindyn: kindyncomputations.KinDynComputations,
              local_foot_vertices_pos: List,
              icub: iCub,
              gazebo: scenario.GazeboSimulator,
              nominal_DS_duration: float = 0.04,
              difference_position_threshold: float = 0.04,
              difference_height_norm_threshold: bool = 0.005) -> "KinematicComputations":
        """Build an instance of KinematicComputations."""

        footsteps_extractor = FootstepsExtractor.build(nominal_DS_duration=nominal_DS_duration,
                                                       difference_position_threshold=difference_position_threshold,
                                                       difference_height_norm_threshold=difference_height_norm_threshold)
        postural_extractor = PosturalExtractor.build()

        return KinematicComputations(kindyn=kindyn,
                                     footsteps_extractor=footsteps_extractor,
                                     postural_extractor=postural_extractor,
                                     local_foot_vertices_pos=local_foot_vertices_pos,
                                     icub=icub,
                                     controlled_joints=icub.joint_names(),
                                     gazebo=gazebo)

    def compute_W_vertices_pos(self) -> List:
        """Compute the feet vertices positions in the world (W) frame."""

        # Retrieve the transformation from the world to the base frame
        world_H_base = self.kindyn.get_world_base_transform()

        # Retrieve front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = self.local_foot_vertices_pos[0]
        FR_vertex_pos = self.local_foot_vertices_pos[1]
        BL_vertex_pos = self.local_foot_vertices_pos[2]
        BR_vertex_pos = self.local_foot_vertices_pos[3]

        # Compute right foot (RF) transform w.r.t. the world (W) frame
        base_H_r_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="r_foot")
        W_H_RF = world_H_base.dot(base_H_r_foot)

        # Get the right-foot vertices positions in the world frame
        W_RFL_vertex_pos_hom = W_H_RF @ np.concatenate((FL_vertex_pos, [1]))
        W_RFR_vertex_pos_hom = W_H_RF @ np.concatenate((FR_vertex_pos, [1]))
        W_RBL_vertex_pos_hom = W_H_RF @ np.concatenate((BL_vertex_pos, [1]))
        W_RBR_vertex_pos_hom = W_H_RF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_RFL_vertex_pos = W_RFL_vertex_pos_hom[0:3]
        W_RFR_vertex_pos = W_RFR_vertex_pos_hom[0:3]
        W_RBL_vertex_pos = W_RBL_vertex_pos_hom[0:3]
        W_RBR_vertex_pos = W_RBR_vertex_pos_hom[0:3]

        # Compute left foot (LF) transform w.r.t. the world (W) frame
        base_H_l_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="l_foot")
        W_H_LF = world_H_base.dot(base_H_l_foot)

        # Get the left-foot vertices positions wrt the world frame
        W_LFL_vertex_pos_hom = W_H_LF @ np.concatenate((FL_vertex_pos, [1]))
        W_LFR_vertex_pos_hom = W_H_LF @ np.concatenate((FR_vertex_pos, [1]))
        W_LBL_vertex_pos_hom = W_H_LF @ np.concatenate((BL_vertex_pos, [1]))
        W_LBR_vertex_pos_hom = W_H_LF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_LFL_vertex_pos = W_LFL_vertex_pos_hom[0:3]
        W_LFR_vertex_pos = W_LFR_vertex_pos_hom[0:3]
        W_LBL_vertex_pos = W_LBL_vertex_pos_hom[0:3]
        W_LBR_vertex_pos = W_LBR_vertex_pos_hom[0:3]

        # Store the positions of both right-foot and left-foot vertices in the world frame
        W_vertices_positions = [W_RFL_vertex_pos, W_RFR_vertex_pos, W_RBL_vertex_pos, W_RBR_vertex_pos,
                                W_LFL_vertex_pos, W_LFR_vertex_pos, W_LBL_vertex_pos, W_LBR_vertex_pos]

        return W_vertices_positions

    def set_initial_support_vertex_and_support_foot(self) -> None:
        """Compute initial support foot and support vertex positions in the world frame, along with the support vertex offset."""

        # Compute support foot position wrt the world frame
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_SF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
        W_H_SF = world_H_base.dot(base_H_SF)
        W_support_foot_pos = W_H_SF[0:3, -1]

        # Compute support vertex position wrt the world frame
        F_support_vertex_pos = self.local_foot_vertices_pos[self.support_vertex]
        F_support_vertex_pos_hom = np.concatenate((F_support_vertex_pos, [1]))
        W_support_vertex_pos_hom = W_H_SF @ F_support_vertex_pos_hom
        W_support_vertex_pos = W_support_vertex_pos_hom[0:3]

        # Set initial support foot and support vertex positions, along with the support vertex offset
        self.support_foot_pos = W_support_foot_pos
        self.support_vertex_pos = W_support_vertex_pos
        self.support_vertex_offset = [W_support_vertex_pos[0], W_support_vertex_pos[1], 0]

    def reset_robot_configuration(self,
                                  joint_positions: List,
                                  base_position: List,
                                  base_quaternion: List) -> None:
        """Reset the robot configuration."""

        world_H_base = numpy.FromNumPy.to_idyntree_transform(
            position=np.array(base_position),
            quaternion=np.array(base_quaternion)).asHomogeneousTransform().toNumPy()

        self.kindyn.set_robot_state(s=joint_positions, ds=np.zeros(len(joint_positions)), world_H_base=world_H_base)

    def reset_visual_robot_configuration(self,
                                         joint_positions: List = None,
                                         base_position: List = None,
                                         base_quaternion: List = None) -> None:
        """Reset the configuration of the robot visualized in the simulator."""

        # Reset joint configuration
        if joint_positions is not None:
            self.icub.to_gazebo().reset_joint_positions(joint_positions, self.controlled_joints)

        # Reset base pose
        if base_position is not None and base_quaternion is not None:
            self.icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        elif base_quaternion is not None:
            self.icub.to_gazebo().reset_base_orientation(base_quaternion)
        elif base_position is not None:
            self.icub.to_gazebo().reset_base_position(base_position)

        # Step the simulator (visualization only)
        self.gazebo.run(paused=True)

    def compute_and_apply_kinematically_feasible_base_position(self,
                                                               joint_positions: List,
                                                               base_quaternion: List) -> List:
        """Compute kinematically-feasible base position and update the robot configuration."""

        # Recompute base position by leg odometry
        kinematically_feasible_base_pos = self.compute_base_position_by_leg_odometry()

        # Update the base position in the robot configuration
        self.reset_robot_configuration(joint_positions=joint_positions,
                                       base_position=kinematically_feasible_base_pos,
                                       base_quaternion=base_quaternion)

        # Update the base position in the configuration of the robot visualized in the simulator
        self.reset_visual_robot_configuration(base_position=kinematically_feasible_base_pos)

        return kinematically_feasible_base_pos

    def compute_base_position_by_leg_odometry(self) -> List:
        """Compute kinematically-feasible base position using leg odometry."""

        # Get the base (B) position in the world (W) frame
        W_pos_B = self.kindyn.get_world_base_transform()[0:3, -1]

        # Get the support vertex position in the world (W) frame
        W_support_vertex_pos = self.support_vertex_pos

        # Get the support vertex orientation in the world (W) frame, defined as the support foot (SF) orientation
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_SF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
        W_H_SF = world_H_base.dot(base_H_SF)
        W_support_vertex_quat = Quaternion.from_matrix(W_H_SF[0:3, 0:3])

        # Compute the transform of the support vertex (SV) in the world (W) frame
        W_H_SV = Transform.from_position_and_quaternion(position=np.asarray(W_support_vertex_pos),
                                                        quaternion=np.asarray(W_support_vertex_quat))

        # Express the base (B) position in the support vertex (SV) reference frame
        SV_H_W = np.linalg.inv(W_H_SV)
        W_pos_B_hom = np.concatenate((W_pos_B, [1]))
        SV_pos_B_hom = SV_H_W @ W_pos_B_hom

        # Express the base (B) position in a reference frame oriented as the world but positioned in the support vertex (SV)
        mixed_H_SV = Transform.from_position_and_quaternion(position=np.asarray([0, 0, 0]),
                                                            quaternion=np.asarray(W_support_vertex_quat))
        mixed_pos_B_hom = mixed_H_SV @ SV_pos_B_hom

        # Convert homogeneous to cartesian coordinates
        mixed_pos_B = mixed_pos_B_hom[0:3]

        # Compute the kinematically-feasible base position, i.e. the base position such that the support
        # vertex remains fixed while the robot configuration changes
        kinematically_feasible_base_position = mixed_pos_B + self.support_vertex_offset

        return kinematically_feasible_base_position

    def update_support_vertex_and_support_foot(self) -> (str, bool, bool):
        """Update the support vertex and the support foot. Also, return boolean variables indicating whether the
        deactivation time of the last footstep needs to be updated (update_footstep_deactivation_time) and whether
        a new footstep needs to be added to the footsteps list (update_footsteps_list)."""

        update_footsteps_list = False

        # Associate feet vertices names to indexes
        vertex_indexes_to_names = {0: "RFL", 1: "RFR", 2: "RBL", 3: "RBR",
                                   4: "LFL", 5: "LFR", 6: "LBL", 7: "LBR"}

        # Retrieve the vertices positions in the world frame
        W_vertices_positions = self.compute_W_vertices_pos()

        # Compute the current support vertex
        vertices_heights = [W_vertex[2] for W_vertex in W_vertices_positions]
        self.support_vertex = np.argmin(vertices_heights)

        # Check whether the deactivation time of the last footstep needs to be updated
        update_footstep_deactivation_time = self.footsteps_extractor.should_update_footstep_deactivation_time(kindyn=self.kindyn)

        # If the support vertex did not change
        if self.support_vertex == self.support_vertex_prev:

            # Update support foot position and support vertex position
            world_H_base = self.kindyn.get_world_base_transform()
            base_H_support_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
            W_H_SF = world_H_base.dot(base_H_support_foot)
            self.support_foot_pos = W_H_SF[0:3, -1]
            self.support_vertex_pos = W_vertices_positions[self.support_vertex]

        # If the support vertex changed
        else:

            # Update the support foot
            if vertex_indexes_to_names[self.support_vertex][0] == "R":
                self.support_foot = "r_foot"
            else:
                self.support_foot = "l_foot"

            # If the support foot changed
            if self.support_foot != self.support_foot_prev:

                # Indicate that a new footstep needs to be added to the footsteps list
                update_footsteps_list = True

                # Update support foot prev
                self.support_foot_prev = self.support_foot

            # Update support foot position and support vertex position
            world_H_base = self.kindyn.get_world_base_transform()
            base_H_support_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.support_foot)
            W_H_SF = world_H_base.dot(base_H_support_foot)
            self.support_foot_pos = W_H_SF[0:3, -1]
            self.support_vertex_pos = W_vertices_positions[self.support_vertex]

            # Update also the vertex offset
            self.support_vertex_offset = [self.support_vertex_pos[0], self.support_vertex_pos[1], 0]

            # Update support vertex prev
            self.support_vertex_prev = self.support_vertex

        return self.support_foot, update_footstep_deactivation_time, update_footsteps_list


@dataclass
class Plotter:
    """Class to handle the plots related to the trajectory generation pipeline."""

    # Axis of the composed ellipsoid constraining the last point of the Bezier curve of base positions
    ellipsoid_forward_axis: float
    ellipsoid_side_axis: float
    ellipsoid_backward_axis: float

    # Scaling factor for all the axes of the composed ellipsoid
    ellipsoid_scaling: float

    @staticmethod
    def build(ellipsoid_forward_axis: float = 1.0,
              ellipsoid_side_axis: float = 0.9,
              ellipsoid_backward_axis: float = 0.6,
              ellipsoid_scaling: float = 0.4) -> "Plotter":
        """Build an instance of Plotter."""

        return Plotter(ellipsoid_forward_axis=ellipsoid_forward_axis,
                       ellipsoid_side_axis=ellipsoid_side_axis,
                       ellipsoid_backward_axis=ellipsoid_backward_axis,
                       ellipsoid_scaling=ellipsoid_scaling)

    @staticmethod
    def plot_blending_coefficients(figure_blending_coefficients: int, blending_coeffs: Dict) -> None:
        """Plot the activations of the blending coefficients."""

        plt.figure(figure_blending_coefficients)
        plt.clf()

        # Plot blending coefficients
        plt.plot(range(len(blending_coeffs["w_1"])), blending_coeffs["w_1"], 'r')
        plt.plot(range(len(blending_coeffs["w_2"])), blending_coeffs["w_2"], 'b')
        plt.plot(range(len(blending_coeffs["w_3"])), blending_coeffs["w_3"], 'g')
        plt.plot(range(len(blending_coeffs["w_4"])), blending_coeffs["w_4"], 'y')

        # Plot configuration
        plt.title("Blending coefficients profiles")
        plt.ylabel("Blending coefficients")
        plt.xlabel("Time [s]")

    @staticmethod
    def plot_new_footstep(figure_footsteps: int, support_foot: str, new_footstep: Dict) -> None:
        """Plot a new footstep just added to the footsteps list."""

        plt.figure(figure_footsteps)

        # Plot left footsteps in blue, right footsteps in red
        colors={"l_foot": 'b', "r_foot": 'r'}

        # Footstep position
        plt.scatter(new_footstep["2D_pos"][1], -new_footstep["2D_pos"][0], c=colors[support_foot])

        # Footstep orientation (scaled for visualization purposes)
        plt.plot([new_footstep["2D_pos"][1], new_footstep["2D_pos"][1] + math.sin(new_footstep["2D_orient"]) / 10],
                 [-new_footstep["2D_pos"][0], -new_footstep["2D_pos"][0] - math.cos(new_footstep["2D_orient"]) / 10],
                 colors[support_foot])

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.title("Footsteps")

    @staticmethod
    def plot_predicted_future_trajectory(figure_base_vel: int, figure_base_ang_vel: int, denormalized_current_output: List) -> None:
        """Plot the future trajectory predicted by the network (magenta)."""

        # Retrieve predicted base velocities and angular velocities from the denormalized network output
        predicted_base_vel = denormalized_current_output[3:21]
        predicted_base_ang_vel = denormalized_current_output[24:42]

        plt.figure(figure_base_vel)

        # Separate x,y,z components
        predicted_vel_x = []
        predicted_vel_y = []
        predicted_vel_z = []
        for k in range(0, len(predicted_base_vel), 3):
            predicted_vel_x.append(predicted_base_vel[k])
            predicted_vel_y.append(predicted_base_vel[k+1])
            predicted_vel_z.append(predicted_base_vel[k+2])

        # Plot linear velocities over time
        plt.scatter(np.linspace(1/6, 1, 6), predicted_vel_x, c='m', label="Predicted x")
        plt.scatter(np.linspace(1/6, 1, 6), predicted_vel_y, c='m', marker="+", label="Predicted y")

        plt.figure(figure_base_ang_vel)

        # Separate x,y,z components
        predicted_ang_vel_x = []
        predicted_ang_vel_y = []
        predicted_ang_vel_z = []
        for k in range(0, len(predicted_base_ang_vel), 3):
            predicted_ang_vel_x.append(predicted_base_ang_vel[k])
            predicted_ang_vel_y.append(predicted_base_ang_vel[k+1])
            predicted_ang_vel_z.append(predicted_base_ang_vel[k+2])

        # Plot angular velocities over time
        plt.scatter(np.linspace(1/6, 1, 6), predicted_ang_vel_z, c='m', marker="^", label="Predicted z")


    @staticmethod
    def plot_desired_future_trajectory(figure_base_vel: int, figure_base_ang_vel: int, base_velocities: List, 
                                       base_angular_velocities: List) -> None:
        """Plot the future trajectory built from user inputs (gray)."""

        plt.figure(figure_base_vel)

        # Separate x,y components
        desired_vel_x = [-elem[1] for elem in base_velocities]
        desired_vel_y = [elem[0] for elem in base_velocities]

        # Plot linear velocities over time
        plt.scatter(np.linspace(0, 1, 7), desired_vel_x, c='gray', label="Desired x")
        plt.scatter(np.linspace(0, 1, 7), desired_vel_y, c='gray', marker="+", label="Desired y")

        plt.figure(figure_base_ang_vel)

        # Plot angular velocities over time
        plt.scatter(np.linspace(0, 1, 7), base_angular_velocities, c='gray', marker="^", label="Desired z")

    @staticmethod
    def plot_blended_future_trajectory(figure_base_vel: int, figure_base_ang_vel: int, blended_base_velocities: List,
                                       blended_base_angular_velocities: List) -> None:
        """Plot the future trajectory obtained by blending the network output and the user input (green)."""

        # Extract components for plotting
        blended_base_velocities_x = [elem[1] for elem in blended_base_velocities]
        blended_base_velocities_y = [-elem[0] for elem in blended_base_velocities]

        plt.figure(figure_base_vel)

        # Plot linear velocities over time
        plt.scatter(np.linspace(0, 1, 7), blended_base_velocities_x, c='g', label="Blended x")
        plt.scatter(np.linspace(0, 1, 7), blended_base_velocities_y, c='g', marker="+", label="Blended y")

        plt.figure(figure_base_ang_vel)

        # Plot angular velocities over time
        plt.scatter(np.linspace(0, 1, 7), blended_base_angular_velocities, c='g', marker="^", label="Blended z")


    def plot_trajectory_blending(self, figure_base_vel: int, figure_base_ang_vel: int, denormalized_current_output: List,
                                 base_velocities: List, base_angular_velocities: List, blended_base_velocities: List,
                                 blended_base_angular_velocities: List) -> None:
        """Plot the predicted, desired and blended future trajectories used to build the next network input."""

        # Base velocities plot
        plt.figure(figure_base_vel)
        plt.clf()

        # Base angular velocities plot
        plt.figure(figure_base_ang_vel)
        plt.clf()

        # Plot the future trajectory predicted by the network
        self.plot_predicted_future_trajectory(figure_base_vel=figure_base_vel, figure_base_ang_vel=figure_base_ang_vel,
                                              denormalized_current_output=denormalized_current_output)

        # Plot the future trajectory built from user inputs
        self.plot_desired_future_trajectory(figure_base_vel=figure_base_vel, figure_base_ang_vel=figure_base_ang_vel, 
                                            base_velocities=base_velocities, 
                                            base_angular_velocities=base_angular_velocities)

        # Plot the future trajectory obtained by blending the network output and the user input
        self.plot_blended_future_trajectory(figure_base_vel=figure_base_vel, figure_base_ang_vel=figure_base_ang_vel,
                                            blended_base_velocities=blended_base_velocities,
                                            blended_base_angular_velocities=blended_base_angular_velocities)

        # Configure base velocities plot
        plt.figure(figure_base_vel)
        plt.xlim([0, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title("BASE VELOCITIES")
        plt.ylim([-0.5, 0.5])
        plt.legend()

        # Configure base angular velocities plot
        plt.figure(figure_base_ang_vel)
        plt.xlim([0, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title("BASE ANGULAR VELOCITIES")
        plt.ylim([-1.0, 1.0])
        plt.legend()


@dataclass
class LearnedModel:
    """Class for the direct exploitation of the model learned during training."""

    # Path to the learned model
    model_path: str

    # Output mean and standard deviation
    Ymean: List
    Ystd: List

    @staticmethod
    def build(training_path: str) -> "LearnedModel":
        """Build an instance of LearnedModel."""

        # Retrieve path to the learned model
        model_path = os.path.join(training_path, "model/")

        # Compute output mean and standard deviation
        datapath = os.path.join(training_path, "normalization/")
        Ymean, Ystd = load_output_mean_and_std(datapath)

        return LearnedModel(model_path=model_path, Ymean=Ymean, Ystd=Ystd)

    def restore_learned_model(self, session: tf.Session) -> tf.Graph:
        """Restore the learned model."""

        # Restore the network generator
        saver = tf.train.import_meta_graph(self.model_path + "/model.ckpt.meta")
        saver.restore(session, tf.train.latest_checkpoint(self.model_path))

        # Retrieve the graph
        graph = tf.get_default_graph()

        return graph

    @staticmethod
    def retrieve_tensors(graph: tf.Graph) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        """Retrieve the tensors associated to the quantities of interest."""

        # Placeholder to feed the input X
        nn_X = graph.get_tensor_by_name("nn_X:0")

        # Placeholder to feed the dropout probability
        nn_keep_prob = graph.get_tensor_by_name("nn_keep_prob:0")

        # Tensor containing the network output
        output = graph.get_tensor_by_name('Squeeze:0')

        # Tensor containing the blending coefficients
        blending_coefficients = graph.get_tensor_by_name('Softmax:0')

        return nn_X, nn_keep_prob, output, blending_coefficients

    @staticmethod
    def evaluate_tensors(nn_X: tf.Tensor, current_nn_X: List, nn_keep_prob: tf.Tensor, output: tf.Tensor,
                         blending_coefficients: tf.Tensor) -> (np.array, np.array):
        """Evaluate the tensors associated to the quantities of interest."""

        # Pass the input defined at the previous iteration to the network (no dropout at inference time)
        feed_dict = {nn_X: current_nn_X, nn_keep_prob: 1.0}

        # Extract the output from the network
        current_output = output.eval(feed_dict=feed_dict)

        # Extract the blending coefficients from the network
        current_blending_coefficients = blending_coefficients.eval(feed_dict=feed_dict)

        return current_output, current_blending_coefficients


@dataclass
class Autoregression:
    """Class to use the network output, blended with the user-specified input, in an autoregressive fashion."""

    # Component-wise input mean and standard deviation
    Xmean_dict: Dict
    Xstd_dict: Dict

    # Blending parameters tau
    tau_base_velocities: float
    tau_base_angular_velocities: float

    # Auxiliary variable to handle unnatural in-place rotations when the robot is stopped
    nn_X_difference_norm_threshold: float

    # Variables to store autoregression-relevant information for the current iteration
    current_nn_X: List
    current_past_trajectory_base_velocities: List
    current_past_trajectory_base_angular_velocities: List
    current_base_position: np.array
    current_base_angle: np.array
    current_world_R_base: np.array = field(default_factory=lambda: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    # Variables to store autoregression-relevant information for the next iteration
    next_nn_X: List = field(default_factory=list)
    new_past_trajectory_base_velocities: List = field(default_factory=list)
    new_past_trajectory_base_angular_velocities: List = field(default_factory=list)
    new_base_position: List = field(default_factory=list)
    new_world_R_base: List = field(default_factory=list)
    new_base_R_world: List = field(default_factory=list)
    new_base_angle: List = field(default_factory=list)

    # Number of points constituting the Bezier curve
    t: List = field(default_factory=lambda: np.linspace(0, 1, 7))

    # Relevant indexes of the window storing past data
    past_window_indexes: List = field(default_factory=lambda: [0, 10, 20, 30, 40, 50])

    # Auxiliary variable for the robot status (moving or stopped)
    stopped: bool = True

    @staticmethod
    def build(training_path: str,
              initial_nn_X: List,
              initial_past_trajectory_base_vel: List,
              initial_past_trajectory_base_ang_vel: List,
              initial_base_height: List,
              initial_base_angle: np.array,
              tau_base_velocities: float = 1.3,
              tau_base_angular_velocities: float = 1.3,
              nn_X_difference_norm_threshold: float = 0.05) -> "Autoregression":
        """Build an instance of Autoregression."""

        # Compute component-wise input mean and standard deviation
        datapath = os.path.join(training_path, "normalization/")
        Xmean_dict, Xstd_dict = load_component_wise_input_mean_and_std(datapath)

        return Autoregression(Xmean_dict=Xmean_dict,
                              Xstd_dict=Xstd_dict,
                              tau_base_velocities=tau_base_velocities,
                              tau_base_angular_velocities=tau_base_angular_velocities,
                              nn_X_difference_norm_threshold=nn_X_difference_norm_threshold,
                              current_nn_X=initial_nn_X,
                              current_past_trajectory_base_velocities=initial_past_trajectory_base_vel,
                              current_past_trajectory_base_angular_velocities=initial_past_trajectory_base_ang_vel,
                              current_base_position=np.array([0, 0, initial_base_height]),
                              current_base_angle=initial_base_angle)

    def update_reference_frame(self, world_H_base: np.array) -> None:
        """Update the local reference frame given by the new base position and orientation."""

        # Store new base position
        self.new_base_position = world_H_base[0:3, -1]

        self.new_world_R_base = world_H_base[:3, :3] 
        self.new_base_R_world = np.linalg.inv(self.new_world_R_base)

    def autoregressive_usage_base_velocities(self, next_nn_X: List, denormalized_current_output: np.array,
                                             base_velocities: List) -> (List, List):
        """Use the base velocities in an autoregressive fashion."""

        # ====================
        # PAST BASE VELOCITIES
        # ====================

        # Update the full window storing the past base velocities
        new_past_trajectory_base_velocities = []
        for k in range(len(self.current_past_trajectory_base_velocities) - 1):
            # Element in the reference frame defined by the previous base position + orientation
            base_elem = self.current_past_trajectory_base_velocities[k + 1]
            # Express element in world frame
            world_elem = self.current_world_R_base.dot(base_elem)
            # Express element in the frame defined by the new base position + orientation
            new_base_elem = self.new_base_R_world.dot(world_elem)
            # Store updated element
            new_past_trajectory_base_velocities.append(new_base_elem)

        # Add as last element the current (local) base velocity (from the output)
        new_past_trajectory_base_velocities.append([denormalized_current_output[0], denormalized_current_output[1], denormalized_current_output[2]])

        # Update past base velocities
        self.new_past_trajectory_base_velocities = new_past_trajectory_base_velocities

        # Extract compressed window of past base velocities (denormalized for plotting)
        past_base_velocities_plot = []
        for index in self.past_window_indexes:
            past_base_velocities_plot.extend(self.new_past_trajectory_base_velocities[index])

        # Extract compressed window of past base velocities (normalized for building the next input)
        past_base_velocities = past_base_velocities_plot.copy()
        for k in range(len(past_base_velocities)):
            past_base_velocities[k] = (past_base_velocities[k] - self.Xmean_dict["past_base_velocities"][k]) / \
                                      self.Xstd_dict["past_base_velocities"][k]

        # Add the compressed window of normalized past ground base velocities to the next input
        next_nn_X.extend(past_base_velocities)

        # ======================
        # FUTURE BASE VELOCITIES
        # ======================

        # Extract future base velocities for blending (i.e. in the plot reference frame)
        future_base_vel_plot = denormalized_current_output[0:21]
        
        future_base_vel_blend = []
        for k in range(0, len(future_base_vel_plot), 3):
            future_base_vel_blend.append([-future_base_vel_plot[k + 1], future_base_vel_plot[k]])

        # Put joystick velocities in terms 
        negated_base_velocities = []
        for k in range(0,len(base_velocities)):
            negated_base_velocities.append([-base_velocities[k][0], -base_velocities[k][1]])

        # Blend user-specified and network-predicted future base velocities
        blended_base_velocities = trajectory_blending(future_base_vel_blend, negated_base_velocities, self.t, self.tau_base_velocities)

        future_base_velocities_blend_features = []
        for k in range(1, len(blended_base_velocities)):
            future_base_velocities_blend_features.append(blended_base_velocities[k][1])
            future_base_velocities_blend_features.append(-blended_base_velocities[k][0])
            future_base_velocities_blend_features.append(future_base_vel_plot[k*3+2])

        # Normalize future base velocities
        future_base_velocities_blend_features_normalized = future_base_velocities_blend_features.copy()
        for k in range(len(future_base_velocities_blend_features_normalized)):
            future_base_velocities_blend_features_normalized[k] = (future_base_velocities_blend_features_normalized[k] -
                                                                   self.Xmean_dict["future_base_velocities"][k]) / \
                                                                  self.Xstd_dict["future_base_velocities"][k]

        # Add the normalized blended future base velocities to the next input
        next_nn_X.extend(future_base_velocities_blend_features_normalized)

        return next_nn_X, blended_base_velocities

    def autoregressive_usage_base_angular_velocities(self, next_nn_X: List, denormalized_current_output: np.array,
                                             base_angular_velocities: List) -> (List, List):
        """Use the base angular velocities in an autoregressive fashion."""

        # ============================
        # PAST BASE ANGULAR VELOCITIES
        # ============================

        # Update the full window storing the past base angular velocities
        new_past_trajectory_base_angular_velocities = []
        for k in range(len(self.current_past_trajectory_base_angular_velocities) - 1):
            # Element in the reference frame defined by the previous base position + orientation
            base_elem = self.current_past_trajectory_base_angular_velocities[k + 1]
            # Express element in world frame
            world_elem = self.current_world_R_base.dot(base_elem)
            # Express element in the frame defined by the new base position + orientation
            new_base_elem = self.new_base_R_world.dot(world_elem)
            # Store updated element
            new_past_trajectory_base_angular_velocities.append(new_base_elem)

        # Add as last element the current (local) base angular velocity (from the output)
        new_past_trajectory_base_angular_velocities.append([denormalized_current_output[21], denormalized_current_output[22], denormalized_current_output[23]])

        # Update past base angular velocities
        self.new_past_trajectory_base_angular_velocities = new_past_trajectory_base_angular_velocities

        # Extract compressed window of past base angular velocities (denormalized for plotting)
        past_base_angular_velocities_plot = []
        for index in self.past_window_indexes:
            past_base_angular_velocities_plot.extend(self.new_past_trajectory_base_angular_velocities[index])

        # Extract compressed window of past base angular velocities (normalized for building the next input)
        past_base_angular_velocities = past_base_angular_velocities_plot.copy()
        for k in range(len(past_base_angular_velocities)):
            past_base_angular_velocities[k] = (past_base_angular_velocities[k] - self.Xmean_dict["past_base_angular_velocities"][k]) / \
                                      self.Xstd_dict["past_base_angular_velocities"][k]

        # Add the compressed window of normalized past ground base angular velocities to the next input
        next_nn_X.extend(past_base_angular_velocities)

        # ==============================
        # FUTURE BASE ANGULAR VELOCITIES
        # ==============================

        # Extract future base angular velocities for blending (i.e. in the plot reference frame)
        future_base_ang_vel_plot = denormalized_current_output[21:42]
        
        future_base_ang_vel_blend = []
        for k in range(0, len(future_base_ang_vel_plot), 3):
            future_base_ang_vel_blend.append(future_base_ang_vel_plot[k+2])

        # Blend user-specified and network-predicted future base angular velocities
        blended_base_angular_velocities = trajectory_blending(future_base_ang_vel_blend, base_angular_velocities, self.t, self.tau_base_angular_velocities)

        future_base_angular_velocities_blend_features = []
        for k in range(1, len(blended_base_angular_velocities)):
            future_base_angular_velocities_blend_features.append(future_base_ang_vel_plot[k*3])
            future_base_angular_velocities_blend_features.append(future_base_ang_vel_plot[k*3+1])
            future_base_angular_velocities_blend_features.append(blended_base_angular_velocities[k])

        # Normalize future base angular velocities
        future_base_ang_velocities_blend_features_normalized = future_base_angular_velocities_blend_features.copy()
        for k in range(len(future_base_ang_velocities_blend_features_normalized)):
            future_base_ang_velocities_blend_features_normalized[k] = (future_base_ang_velocities_blend_features_normalized[k] -
                                                                   self.Xmean_dict["future_base_angular_velocities"][k]) / \
                                                                  self.Xstd_dict["future_base_angular_velocities"][k]

        # Add the normalized blended future base angular velocities to the next input
        next_nn_X.extend(future_base_ang_velocities_blend_features_normalized)

        return next_nn_X, blended_base_angular_velocities


    def autoregressive_usage_joint_positions_and_velocities(self, next_nn_X: List, current_output: np.array) -> List:
        """Use the joint positions and velocities in an autoregressive fashion."""

        # Add the (already normalized) joint positions to the next input
        s = current_output[0][42:74]
        next_nn_X.extend(s)

        # Add the (already normalized) joint velocities to the next input
        s_dot = current_output[0][74:106]
        next_nn_X.extend(s_dot)

        return next_nn_X

    def check_robot_stopped(self, next_nn_X: List) -> None:
        """Check whether the robot is stopped (i.e. whether subsequent network inputs are almost identical)."""

        # Compute the difference in norm between the current and the next network inputs
        nn_X_difference_norm = np.linalg.norm(np.array(self.current_nn_X[0]) - np.array(next_nn_X))

        # The robot is considered to be stopped if the difference in norm is lower than a threshold
        if nn_X_difference_norm < self.nn_X_difference_norm_threshold:
            self.stopped = True
        else:
            self.stopped = False

    def update_autoregression_state(self, next_nn_X: List) -> None:
        """Update the autoregression-relevant information."""

        self.current_nn_X = [next_nn_X]
        self.current_past_trajectory_base_velocities = self.new_past_trajectory_base_velocities
        self.current_past_trajectory_base_angular_velocities = self.new_past_trajectory_base_angular_velocities
        self.current_world_R_base = self.new_world_R_base
        self.current_base_position = self.new_base_position
        self.current_base_angle = self.new_base_angle

    def autoregression_and_blending(self, current_output: np.array, denormalized_current_output: np.array,
                                    base_velocities: List, base_angular_velocities: List,
                                    world_H_base: np.array) -> (List, List):
        """Handle the autoregressive usage of the network output blended with the user input from the joystick."""

        # Update the reference frame given by the base position and orientation
        self.update_reference_frame(world_H_base=world_H_base)

        # Initialize empty next input
        next_nn_X = []

        # Use the base velocities in an autoregressive fashion
        next_nn_X, blended_base_velocities = \
            self.autoregressive_usage_base_velocities(next_nn_X=next_nn_X,
                                                      denormalized_current_output=denormalized_current_output,
                                                      base_velocities=base_velocities)

        # Use the base angular velocities in an autoregressive fashion
        next_nn_X, blended_base_angular_velocities = \
            self.autoregressive_usage_base_angular_velocities(next_nn_X=next_nn_X,
                                                      denormalized_current_output=denormalized_current_output,
                                                      base_angular_velocities=base_angular_velocities)

        # Use the joint positions and velocities in an autoregressive fashion
        next_nn_X = self.autoregressive_usage_joint_positions_and_velocities(next_nn_X, current_output)

        # Check whether the robot is stopped
        self.check_robot_stopped(next_nn_X)

        # Update autoregressive-relevant information for the next iteration
        self.update_autoregression_state(next_nn_X)

        return blended_base_velocities, blended_base_angular_velocities


@dataclass
class TrajectoryGenerator:
    """Class for generating trajectories."""

    # Subcomponents of the trajectory generator
    kincomputations: KinematicComputations
    storage: StorageHandler
    autoregression: Autoregression
    plotter: Plotter
    model: LearnedModel

    # Iteration counter and generation rate
    iteration: int = 0
    generation_rate: float = 1/50

    @staticmethod
    def build(icub: iCub,
              gazebo: scenario.GazeboSimulator,
              kindyn: kindyncomputations.KinDynComputations,
              storage_path: str,
              training_path: str,
              local_foot_vertices_pos: List,
              initial_nn_X: List,
              initial_past_trajectory_base_vel: List,
              initial_past_trajectory_base_ang_vel: List,
              initial_base_height: List,
              initial_base_angle: np.array,
              nominal_DS_duration: float = 0.04,
              difference_position_threshold: float = 0.04,
              difference_height_norm_threshold: bool = 0.005,
              tau_base_velocities: float = 1.3,
              tau_base_angular_velocities: float = 1.3,
              nn_X_difference_norm_threshold: float = 0.05,
              ellipsoid_forward_axis: float = 1.0,
              ellipsoid_side_axis: float = 0.9,
              ellipsoid_backward_axis: float = 0.6,
              ellipsoid_scaling: float = 0.4) -> "TrajectoryGenerator":
        """Build an instance of TrajectoryGenerator."""

        # Build the kinematic computations handler component
        kincomputations = KinematicComputations.build(kindyn=kindyn,
                                                      local_foot_vertices_pos=local_foot_vertices_pos,
                                                      icub=icub,
                                                      gazebo=gazebo,
                                                      nominal_DS_duration=nominal_DS_duration,
                                                      difference_position_threshold=difference_position_threshold,
                                                      difference_height_norm_threshold=difference_height_norm_threshold)

        # Initialize the support vertex and the support foot
        kincomputations.set_initial_support_vertex_and_support_foot()

        # Build the storage handler component
        storage = StorageHandler.build(storage_path)

        # Build the autoregression handler component
        autoregression = Autoregression.build(training_path=training_path,
                                              initial_nn_X=initial_nn_X,
                                              initial_past_trajectory_base_vel=initial_past_trajectory_base_vel,
                                              initial_past_trajectory_base_ang_vel=initial_past_trajectory_base_ang_vel,
                                              initial_base_height=initial_base_height,
                                              initial_base_angle=initial_base_angle,
                                              tau_base_velocities=tau_base_velocities,
                                              tau_base_angular_velocities=tau_base_angular_velocities,
                                              nn_X_difference_norm_threshold=nn_X_difference_norm_threshold)

        # Build the plotter component
        plotter = Plotter.build(ellipsoid_forward_axis=ellipsoid_forward_axis,
                                ellipsoid_side_axis=ellipsoid_side_axis,
                                ellipsoid_backward_axis=ellipsoid_backward_axis,
                                ellipsoid_scaling=ellipsoid_scaling)

        # Build the learned model component
        model = LearnedModel.build(training_path=training_path)

        return TrajectoryGenerator(kincomputations=kincomputations,
                                   storage=storage,
                                   autoregression=autoregression,
                                   plotter=plotter,
                                   model=model)

    def restore_model_and_retrieve_tensors(self, session: tf.Session) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        """Restore the learned model and retrieve the tensors of interest."""

        # Restore the learned model
        graph = self.model.restore_learned_model(session=session)

        # Retrieve the tensors of interest
        nn_X, nn_keep_prob, output, blending_coefficients = self.model.retrieve_tensors(graph)

        return nn_X, nn_keep_prob, output, blending_coefficients

    def retrieve_network_output_and_blending_coefficients(self, nn_X: tf.Tensor, nn_keep_prob: tf.Tensor, output: tf.Tensor,
                                                          blending_coefficients: tf.Tensor) -> (np.array, np.array, np.array):
        """Retrieve the network output (also denormalized) and the blending coefficients."""

        # Retrieve the network output and the blending coefficients
        current_output, current_blending_coefficients = self.model.evaluate_tensors(nn_X=nn_X,
                                                                                    current_nn_X=self.autoregression.current_nn_X,
                                                                                    nn_keep_prob=nn_keep_prob,
                                                                                    output=output,
                                                                                    blending_coefficients=blending_coefficients)

        # Denormalize the network output
        denormalized_current_output = denormalize(current_output, self.model.Ymean, self.model.Ystd)[0]

        return current_output, denormalized_current_output, current_blending_coefficients

    def apply_joint_positions_and_base_orientation(self, denormalized_current_output: List) -> (List, List):
        """Apply joint positions and base orientation from the output returned by the network."""

        # Extract the new joint positions from the denormalized network output
        joint_positions = np.asarray(denormalized_current_output[42:74])

        # If the robot is stopped, handle unnatural in-place rotations by imposing zero angular base velocity
        if self.autoregression.stopped:
            base_angular_velocity = np.array([0.0,0.0,0.0])
        else:
            base_angular_velocity = np.array(denormalized_current_output[21:24])

        # Extract the new base orientation from the output
        base_angle_change = base_angular_velocity * self.generation_rate
        new_base_angle = self.autoregression.current_base_angle + base_angle_change
        new_base_rotation = Rotation.from_euler('xyz', [0, 0, new_base_angle[2]])
        new_base_quaternion = Quaternion.to_wxyz(new_base_rotation.as_quat())

        # Update the base orientation and the joint positions in the robot configuration
        self.kincomputations.reset_robot_configuration(joint_positions=joint_positions,
                                                       base_position=self.autoregression.current_base_position,
                                                       base_quaternion=new_base_quaternion)

        # Update the base base orientation and the joint positions in the configuration of the robot visualized in the simulator
        self.kincomputations.reset_visual_robot_configuration(joint_positions=joint_positions,
                                                              base_quaternion=new_base_quaternion)

        # Update the base yaw in the autoregression state
        self.autoregression.new_base_angle = new_base_angle

        return joint_positions, new_base_quaternion

    def update_support_vertex_and_support_foot_and_footsteps(self) -> (str, bool):
        """Update the support vertex and the support foot. Handle updates of the footsteps list and of the deactivation
        time of the last footstep."""

        # Update support foot and support vertex while detecting new footsteps and deactivation time updates
        support_foot, update_deactivation_time, update_footsteps_list = self.kincomputations.update_support_vertex_and_support_foot()

        if update_deactivation_time:

            # Define the swing foot
            if support_foot == "r_foot":
                swing_foot = "l_foot"
            else:
                swing_foot = "r_foot"

            if self.storage.footsteps[swing_foot]:

                # Update the deactivation time of the last footstep
                self.storage.footsteps[swing_foot][-1]["deactivation_time"] = self.iteration * self.generation_rate

        if update_footsteps_list:

            # Retrieve the information related to the new footstep
            new_footstep = self.kincomputations.footsteps_extractor.create_new_footstep(
                kindyn=self.kincomputations.kindyn,
                support_foot=support_foot,
                activation_time=self.iteration * self.generation_rate)

            # Update the footsteps storage
            self.storage.update_footsteps_storage(support_foot=support_foot, footstep=new_footstep)

        return support_foot, update_footsteps_list

    def compute_kinematically_feasible_base_and_update_posturals(self, joint_positions: List,
                                                                base_quaternion: List, controlled_joints: List,
                                                                link_names: List) -> (List, List, List, List):
        """Compute kinematically-feasible base position and retrieve updated posturals."""

        # Compute and apply kinematically-feasible base position
        kinematically_feasible_base_position = \
            self.kincomputations.compute_and_apply_kinematically_feasible_base_position( joint_positions=joint_positions,
                                                                                         base_quaternion=base_quaternion)

        # Retrieve new posturals to be added to the list of posturals
        new_base_postural, new_joints_postural, new_links_postural, new_com_postural = \
            self.kincomputations.postural_extractor.create_new_posturals(base_position=kinematically_feasible_base_position,
                                                                         base_quaternion=base_quaternion,
                                                                         joint_positions=joint_positions,
                                                                         controlled_joints=controlled_joints,
                                                                         kindyn=self.kincomputations.kindyn,
                                                                         link_names=link_names)

        return new_base_postural, new_joints_postural, new_links_postural, new_com_postural

    def retrieve_joystick_inputs(self, input_port: yarp.BufferedPortBottle, base_velocities: List,
                                 base_angular_velocities: List, raw_data: List) -> (List, List, List):
        """Retrieve user-specified joystick inputs received through YARP port."""

        # The joystick input from the user written on the YARP port will contain 2 * 7 + 7 + 4 = 25 values:
        # 0-13 are base_velocities (x,y)
        # 14-20 are base_angular_velocities (z)
        # 21-24 are joystick inputs to be stored for future plotting (curr_x, curr_y, curr_z, curr_rz)

        # Read from the input port
        res = input_port.read(shouldWait=False)

        if res is None:

            if base_velocities:

                # If the port is empty but the previous joystick inputs are not empty, return them
                return base_velocities, base_angular_velocities, raw_data

            else:

                # If the port is empty and the previous joystick inputs are empty, return default values
                default_base_velocities = [[0, 0] for _ in range(len(self.autoregression.t))]
                default_base_angular_velocities = [0 for _ in range(len(self.autoregression.t))]
                default_raw_data = [0, 0, 0, -1] # zero motion direction (robot stopped), forward base direction

                return default_base_velocities, default_base_angular_velocities, default_raw_data

        else:

            # If the port is not empty, retrieve the new joystick inputs
            new_base_velocities = []
            new_base_angular_velocities = []
            new_raw_data = []

            for k in range(0, 13, 2):
                coords = [res.get(k).asFloat32(), res.get(k + 1).asFloat32()]
                new_base_velocities.append(coords)

            for k in range(14, res.size()):
                if k < res.size() - 4:
                    new_base_angular_velocities.append(res.get(k).asFloat32())
                else:
                    new_raw_data.append(res.get(k).asFloat32())

            return new_base_velocities, new_base_angular_velocities, new_raw_data

    def autoregression_and_blending(self, current_output: np.array, denormalized_current_output: np.array, base_velocities: List,
                                    base_angular_velocities: List) -> (List, List):
        """Use the network output in an autoregressive fashion and blend it with the user input."""

        world_H_base = self.kincomputations.kindyn.get_world_base_transform()

        # Use the network output in an autoregressive fashion and blend it with the user input
        blended_base_velocities, blended_base_angular_velocities = \
            self.autoregression.autoregression_and_blending(current_output=current_output,
                                                            denormalized_current_output=denormalized_current_output,
                                                            base_velocities=base_velocities,
                                                            base_angular_velocities=base_angular_velocities,
                                                            world_H_base=world_H_base)

        return blended_base_velocities, blended_base_angular_velocities

    def update_storages_and_save(self, blending_coefficients: List, base_postural: List, joints_postural: List,
                                 links_postural: List, com_postural: List, raw_data: List,
                                 base_velocities: List, base_angular_velocities: List, save_every_N_iterations: int) -> None:
        """Update the blending coefficients, posturals and joystick input storages and periodically save data."""

        # Update the blending coefficients storage
        self.storage.update_blending_coefficients_storage(blending_coefficients=blending_coefficients)

        # Update the posturals storage
        self.storage.update_posturals_storage(base=base_postural, joints=joints_postural,
                                              links=links_postural, com=com_postural)

        # Update joystick inputs storage
        self.storage.update_joystick_inputs_storage(raw_data=raw_data, base_velocities=base_velocities, 
                                                    base_angular_velocities=base_angular_velocities)

        # Periodically save data
        if self.iteration % save_every_N_iterations == 0:

            # Before saving data, update the footsteps list
            final_deactivation_time = self.iteration * self.generation_rate
            updated_footsteps = self.kincomputations.footsteps_extractor.update_footsteps(
                final_deactivation_time=final_deactivation_time, footsteps=self.storage.footsteps)
            self.storage.replace_footsteps_storage(footsteps=updated_footsteps)

            # Save data
            self.storage.save_data_as_json()

    def update_iteration_counter(self) -> None:
        """Update the counter for the iterations of the generator."""

        # Debug
        print(self.iteration)
        if self.iteration == 1:
            input("\nPress Enter to start the trajectory generation.\n")

        self.iteration += 1
