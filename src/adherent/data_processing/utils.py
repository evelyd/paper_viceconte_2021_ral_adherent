# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
import json
import numpy as np
from typing import List
from scenario import core
from scenario import gazebo as scenario
from gym_ignition.rbd.idyntree.inverse_kinematics_nlp import IKSolution
from gym_ignition.rbd.idyntree import kindyncomputations
from gym_ignition.rbd.conversions import Quaternion


import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =====================
# MODEL INSERTION UTILS
# =====================

class iCub(core.Model):
    """Helper class to simplify model insertion."""

    def __init__(self,
                 world: scenario.World,
                 urdf: str,
                 position: List[float] = (0., 0, 0),
                 orientation: List[float] = (1., 0, 0, 0)):

        # Insert the model in the world
        name = "iCub"
        pose = core.Pose(position, orientation)
        world.insert_model(urdf, pose, name)

        # Get and store the model from the world
        self.model = world.get_model(model_name=name)

    def __getattr__(self, name):
        return getattr(self.model, name)

# =================
# RETARGETING UTILS
# =================

def define_robot_to_target_base_quat(robot: str) -> List:
    """Define the robot-specific quaternions from the robot base frame to the target base frame."""

    if robot != "iCubV2_5":
        raise Exception("Quaternions from the robot to the target base frame only defined for iCubV2_5.")

    # For iCubV2_5, the robot base frame is rotated of -180 degs on z w.r.t. the target base frame
    robot_to_target_base_quat = [0, 0, 0, -1.0]

    return robot_to_target_base_quat

def define_foot_vertices(robot: str) -> List:
    """Define the robot-specific positions of the feet vertices in the foot frame."""

    if robot != "iCubV2_5":
        raise Exception("Feet vertices positions only defined for iCubV2_5.")

    # For iCubV2_5, the feet vertices are not symmetrically placed wrt the foot frame origin.
    # The foot frame has z pointing down, x pointing forward and y pointing right.

    # Origin of the box which represents the foot (in the foot frame)
    box_origin = [0.03, 0.005, 0.014]

    # Size of the box which represents the foot
    box_size = [0.16, 0.072, 0.001]

    # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
    FL_vertex_pos = [box_origin[0] + box_size[0]/2, box_origin[1] - box_size[1]/2, box_origin[2]]
    FR_vertex_pos = [box_origin[0] + box_size[0]/2, box_origin[1] + box_size[1]/2, box_origin[2]]
    BL_vertex_pos = [box_origin[0] - box_size[0]/2, box_origin[1] - box_size[1]/2, box_origin[2]]
    BR_vertex_pos = [box_origin[0] - box_size[0]/2, box_origin[1] + box_size[1]/2, box_origin[2]]

    # Vertices positions in the foot (F) frame
    F_vertices_pos = [FL_vertex_pos, FR_vertex_pos, BL_vertex_pos, BR_vertex_pos]

    return F_vertices_pos

def quaternion_multiply(quat1: List, quat2: List) -> np.array:
    """Auxiliary function for quaternion multiplication."""

    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    res = np.array([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                     x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                     -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                     x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2])

    return res

def to_xyzw(wxyz: List) -> List:
    """Auxiliary function to convert quaternions from wxyz to xyzw format."""

    return wxyz[[1, 2, 3, 0]]

def store_retargeted_mocap_as_json(timestamps: List, ik_solutions: List, outfile_name: str) -> None:
    """Auxiliary function to store the retargeted motion."""

    ik_solutions_json = []

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        ik_solution_json = {"joint_positions": ik_solution.joint_configuration.tolist(),
                            "base_position": ik_solution.base_position.tolist(),
                            "base_quaternion": ik_solution.base_quaternion.tolist(),
                            "timestamp": timestamps[i]}

        ik_solutions_json.append(ik_solution_json)

    with open(outfile_name, "w") as outfile:
        json.dump(ik_solutions_json, outfile)

def load_retargeted_mocap_from_json(input_file_name: str, initial_frame: int = 0, final_frame: int = -1) -> (List, List):
    """Auxiliary function to load the retargeted mocap data."""

    # Load ik solutions
    with open(input_file_name, 'r') as openfile:
        ik_solutions = json.load(openfile)

    # If a final frame has been passed, extract relevant ik solutions
    if initial_frame != -1:
        ik_solutions = ik_solutions[initial_frame:final_frame]

    # Extract timestamps
    timestamps = [ik_solution["timestamp"] for ik_solution in ik_solutions]

    return timestamps, ik_solutions

# =========================
# FEATURES EXTRACTION UTILS
# =========================

def define_frontal_base_direction(robot: str) -> List:
    """Define the robot-specific frontal base direction in the base frame."""

    if robot != "iCubV2_5":
        raise Exception("Frontal base direction only defined for iCubV2_5.")

    # For iCubV2_5, the reversed x axis of the base frame is pointing forward
    frontal_base_direction = [-1, 0, 0]

    return frontal_base_direction

def define_frontal_chest_direction(robot: str) -> List:
    """Define the robot-specific frontal chest direction in the chest frame."""

    if robot != "iCubV2_5":
        raise Exception("Frontal chest direction only defined for iCubV2_5.")

    # For iCubV2_5, the z axis of the chest frame is pointing forward
    frontal_base_direction = [0, 0, 1]

    return frontal_base_direction

def rotation_2D(angle: float) -> np.array:
    """Auxiliary function for a 2-dimensional rotation matrix."""

    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_retargeted_motion(timestamps: List,
                                ik_solutions: List,
                                icub: iCub,
                                controlled_joints: List,
                                gazebo: scenario.GazeboSimulator) -> None:
    """Auxiliary function to visualize retargeted motion."""

    timestamp_prev = -1

    for i in range(1, len(ik_solutions)):
        print(i, "/", len(ik_solutions))
        ik_solution = ik_solutions[i]

        # Retrieve the base pose and the joint positions, based on the type of ik_solution
        if type(ik_solution) == IKSolution:
            joint_positions = ik_solution.joint_configuration
            base_position = ik_solution.base_position
            base_quaternion = ik_solution.base_quaternion
        elif type(ik_solution) == dict:
            joint_positions = ik_solution["joint_positions"]
            base_position = ik_solution["base_position"]
            base_quaternion = ik_solution["base_quaternion"]

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Visualize the retargeted motion at the time rate of the collected data
        timestamp = timestamps[i]
        if timestamp_prev == -1:
            dt = 1 / 100
        else:
            dt = timestamp - timestamp_prev
        time.sleep(dt)
        timestamp_prev = timestamp

    print("Visualization ended")
    time.sleep(1)

def visualize_candidate_features(ik_solutions: List,
                            icub: iCub,
                            kindyn: kindyncomputations.KinDynComputations,
                            world: scenario.World,
                            controlled_joints: List,
                            gazebo: scenario.GazeboSimulator) -> None:

    print(len(ik_solutions))

    dt_mean = 1/50
    frontal_base_dir = define_frontal_base_direction(robot="iCubV2_5")

    base_heights = np.empty(len(ik_solutions)-1)
    comxs = np.empty(len(ik_solutions)-1)
    comzs = np.empty(len(ik_solutions)-1)
    head_xs = np.empty(len(ik_solutions)-1)
    head_zs = np.empty(len(ik_solutions)-1)
    xz_base_directions = []
    xz_base_directions.append([0.,0.])
    cumulative_pitch = np.empty(len(ik_solutions))
    cumulative_pitch[0] = 0.0
    cumulative_yaw = np.empty(len(ik_solutions))
    cumulative_yaw[0] = 0.0
    ground_base_directions = []
    ground_base_directions.append([0.,0.])
    for i in range(1,len(ik_solutions)):
        # =================
        # BASE HEIGHTS
        # =================
        ik_solution = ik_solutions[i]
        print(i)

        # Retrieve the base pose and the joint positions
        base_position = np.asarray(ik_solution["base_position"])
        base_heights[i-1] = base_position[2]

        # =================
        # COM POSITIONS
        # =================

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)
        #go through and get com for each step
        kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))
        com = kindyn.get_com_position()
        comzs[i-1] = com[2]
        
        #get CoM x in local base frame
        # Get ground rotation from world to base frame
        world_H_base = kindyn.get_world_base_transform()
        # Retrieve the rotation from the facing direction to the world frame and its inverse
        # Express CoM x,y locally
        T_world_to_base = np.linalg.inv(world_H_base)
        current_local_com_pos = T_world_to_base.dot([com[0],com[1],com[2],1])
        comxs[i-1] = current_local_com_pos[0]
        
        # =================
        # HEAD POSITIONS
        # =================
        # Compute head height wrt the world frame
        base_H_head = kindyn.get_relative_transform(ref_frame_name="root_link", frame_name="head")
        W_H_head = world_H_base.dot(base_H_head)
        W_head_ht = W_H_head[2, -1]
        head_zs[i-1] = W_head_ht

        #Compute head x in local base frame
        current_local_head_pos = T_world_to_base.dot([W_H_head[0, -1],W_H_head[1, -1],W_H_head[2, -1],1])
        head_xs[i-1] = current_local_head_pos[0]

        # =================
        # BASE PITCH VELOCITIES
        # =================
        # xz plane base direction
        base_rotation = Quaternion.to_rotation(np.array(base_quaternion))
        base_direction = base_rotation.dot(frontal_base_dir) # we are interested in the frontal base direction
        xz_base_direction = [base_direction[0], base_direction[2]] # project on the xz plane
        xz_base_direction = xz_base_direction / np.linalg.norm(xz_base_direction) # of unitary norm
        xz_base_directions.append(xz_base_direction)

        # Base pitch angular velocities by differentiation of xz base directions
        xz_base_direction_prev = xz_base_directions[-2]
        cos_phi = np.dot(xz_base_direction_prev, xz_base_direction) # unitary norm vectors
        sin_phi = np.cross(xz_base_direction_prev, xz_base_direction) # unitary norm vectors
        phi = math.atan2(sin_phi, cos_phi)
        cumulative_pitch[i] = cumulative_pitch[i-1] + phi
        # xz_base_angular_velocity = phi / dt_mean
        # pitch_vels[i-1] = xz_base_angular_velocity

        # =================
        # BASE YAW VELOCITIES
        # =================
        # ground plane base direction
        ground_base_direction = [base_direction[0], base_direction[1]] # project on the ground plane
        ground_base_direction = ground_base_direction / np.linalg.norm(ground_base_direction) # of unitary norm
        ground_base_directions.append(ground_base_direction)

        # Base yaw angular velocities by differentiation of ground base directions
        ground_base_direction_prev = ground_base_directions[-2]
        cos_theta = np.dot(ground_base_direction_prev, ground_base_direction) # unitary norm vectors
        sin_theta = np.cross(ground_base_direction_prev, ground_base_direction) # unitary norm vectors
        theta = math.atan2(sin_theta, cos_theta)
        cumulative_yaw[i] = cumulative_yaw[i-1] + theta
        # ground_base_angular_velocity = theta / dt_mean
        # yaw_vels[i-1] = ground_base_angular_velocity

    #make plots larger
    plt.rcParams['figure.figsize'] = [16,10]

    # Figure 1 for base heights
    plt.figure(1)
    plt.clf()

    #Plot base heights
    plt.plot(range(1,len(ik_solutions)), base_heights, c='k', label='Base height')
    #Plot standing points (D4 portion 19, mixed walking)
    # xcoords = [1,770,2200,3300,6300,10700,11600,14100,16700,17900,19600,20200,23000,24400,27000,28100,30300,32400]
    #Plot standing points (D4 portion 12, forward walking)
    xcoords = [1,600,1600,2100,3100,4200,5900,6200,6500,9000,10000,10900,12100,15000,16200,16800,18700,19600,20400,21100,22700,23300,23800,24400,25400]
    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('Base heights for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Global base height (m)')
    plt.legend()
    plt.savefig('base_heights_D4_12.png')

    # Figure 2 for CoM x position (local)
    plt.figure(2)
    plt.clf()

    #Plot CoM x
    plt.plot(range(1,len(ik_solutions)), comxs, c='k', label='CoM x')

    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('CoM x for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Local (measured from base) CoM x (m)')
    plt.legend()    
    plt.savefig('com_x_D4_12.png')

    # Figure 3 for CoM z position
    plt.figure(3)
    plt.clf()

    #Plot CoM z
    plt.plot(range(1,len(ik_solutions)), comzs, c='k', label='CoM z')
    #Plot standing points (D4 portion 12, mixed walking)
    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('CoM z for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Global CoM z (m)')
    plt.legend()
    plt.savefig('com_z_D4_12.png')

    # Figure 4 for head x
    plt.figure(4)
    plt.clf()

    #Plot head x
    plt.plot(range(1,len(ik_solutions)), head_xs, c='k', label='Head x')

    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('Head x for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Local (measured from base) head x (m)')
    plt.legend()
    plt.savefig('head_x_D4_12.png')

    # Figure 5 for head z
    plt.figure(5)
    plt.clf()

    #Plot head z
    plt.plot(range(1,len(ik_solutions)), head_zs, c='k', label='Head z')

    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('Head z for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Global head z (m)')
    plt.legend()
    plt.savefig('head_z_D4_12.png')

    # Figure 6 for base pitch
    plt.figure(6)
    plt.clf()

    #Plot base pitch
    plt.plot(range(1,len(ik_solutions)+1), cumulative_pitch, c='k', label='Cumulative base pitch')

    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('Cumulative base pitch for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Base pitch (rad))')
    plt.legend()
    plt.savefig('cumulative_base_pitch_D4_12.png')

    # Figure 7 for base yaw
    plt.figure(7)
    plt.clf()

    #Plot base yaw
    plt.plot(range(1,len(ik_solutions)+1), cumulative_yaw, c='k', label='Cumulative base yaw')

    for xc in xcoords:
        if xc == 1:
            plt.axvline(x=xc, linestyle='--', label='Standing points')
        else:
            plt.axvline(x=xc, linestyle='--')

    plt.grid()
    plt.title('Cumulative base yaw for mixed walking (D4 portion 12)')
    plt.xlabel('Timestep')
    plt.ylabel('Base yaw (rad)')
    plt.legend()
    plt.savefig('cumulative_base_yaw_D4_12.png')

    # #Figure 8 for comparisons of transitions between crouching and walking upright
    # plt.figure(8)
    # plt.clf()

    # #create subplots where x and z plots are next to each other
    # xcoords = [700,800,850,925]
    # plt.subplot(2,4,1)
    # plt.plot(range(400,1000), head_xs[400:1000], c='k', label='Head x')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.ylabel('Global head x (m)')
    # plt.xticks(rotation = 45)
    # plt.grid()
    # plt.subplot(2,4,5)
    # plt.plot(range(400,1000), head_zs[400:1000], c='b', label='Head z')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xlabel('Timestep')
    # plt.ylabel('Global head z (m)')
    # plt.xticks(rotation = 45)
    # plt.grid()

    # xcoords = [6200,6270,6350,6425]
    # plt.subplot(2,4,2)
    # plt.plot(range(6000,6600), head_xs[6000:6600], c='k', label='Head x')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xticks(rotation = 45)
    # plt.grid()
    # plt.subplot(2,4,6)
    # plt.plot(range(6000,6600), head_zs[6000:6600], c='b', label='Head z')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xlabel('Timestep')
    # plt.xticks(rotation = 45)
    # plt.grid()

    # xcoords = [23000,23075,23150,23225]
    # plt.subplot(2,4,3)
    # plt.plot(range(22700,23300), head_xs[22700:23300], c='k', label='Head x')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xticks(rotation = 45)
    # plt.grid()
    # plt.subplot(2,4,7)
    # plt.plot(range(22700,23300), head_zs[22700:23300], c='b', label='Head z')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xlabel('Timestep')
    # plt.xticks(rotation = 45)
    # plt.grid()

    # xcoords = [28070,28150,28200,28300]
    # plt.subplot(2,4,4)
    # plt.plot(range(27800,28400), head_xs[27800:28400], c='k', label='Head x')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xticks(rotation = 45)
    # plt.grid()
    # plt.subplot(2,4,8)
    # plt.plot(range(27800,28400), head_zs[27800:28400], c='b', label='Head z')
    # for xc in xcoords:
    #     if xc == 1:
    #         plt.axvline(x=xc, linestyle='--', label='Transition bounds')
    #     else:
    #         plt.axvline(x=xc, linestyle='--')
    # plt.xlabel('Timestep')
    # plt.xticks(rotation = 45)
    # plt.grid()

    # plt.suptitle('Head x,z transitions for mixed walking (D4 portion 19)')

    # plt.savefig('head_transitions_D4_19.png')

    # Plot
    plt.show()
    plt.pause(0.0001)

def visualize_global_features(global_window_features,
                              ik_solutions: List,
                              icub: iCub,
                              controlled_joints: List,
                              gazebo: scenario.GazeboSimulator,
                              plot_facing_directions: bool = True,
                              plot_base_velocities: bool = False) -> None:
    """Visualize the retargeted frames along with the associated global features."""

    window_length_frames = global_window_features.window_length_frames
    window_step = global_window_features.window_step
    window_indexes = global_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    for i in range(initial_frame, final_frame):

        # Debug
        print(i - initial_frame, "/", final_frame - initial_frame)

        # The ik solutions are stored at double frequency w.r.t. the extracted features
        ik_solution = ik_solutions[2 * i]

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_position = np.asarray(ik_solution["base_position"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Retrieve global features
        base_positions = global_window_features.base_positions[i - window_length_frames]
        facing_directions = global_window_features.facing_directions[i - window_length_frames]
        base_velocities = global_window_features.base_velocities[i - window_length_frames]

        # =================
        # FACING DIRECTIONS
        # =================

        if plot_facing_directions:

            # Figure 1 for the facing directions
            plt.figure(1)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(base_position[1], -base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(base_position[1], -base_position[0], c='k')

                # Plot facing directions
                facing_direction = facing_directions[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current facing direction in blue
                    plt.plot([base_position[1], base_position[1] + 2 * facing_direction[1]],
                             [-base_position[0], -base_position[0] - 2 * facing_direction[0]], 'b',
                             label="Current facing direction")
                else:
                    # Other facing directions in green
                    plt.plot([base_position[1], base_position[1] + facing_direction[1]],
                             [-base_position[0], -base_position[0] - facing_direction[0]], 'g')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-1.75, 1.75])
            plt.ylim([-1.75, 1.75])
            plt.title("Facing directions (global view)")
            plt.legend()

        # ===============
        # BASE VELOCITIES
        # ===============

        if plot_base_velocities:

            # Figure 2 for the base velocities
            plt.figure(2)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(base_position[1], -base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(base_position[1], -base_position[0], c='k')

                # Plot base velocities
                base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current base velocity in magenta
                    plt.plot([base_position[1], base_position[1] + base_velocity[1]],
                             [-base_position[0], -base_position[0] - base_velocity[0]], 'm',
                             label="Current base velocity")
                else:
                    # Other base velocities in gray
                    plt.plot([base_position[1], base_position[1] + base_velocity[1]],
                             [-base_position[0], -base_position[0] - base_velocity[0]], 'gray')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-1.75, 1.75])
            plt.ylim([-1.75, 1.75])
            plt.title("Base velocities (global view)")
            plt.legend()

        # Plot
        plt.show()
        plt.pause(0.0001)

def visualize_local_features(local_window_features,
                             ik_solutions: List,
                             icub: iCub,
                             controlled_joints: List,
                             gazebo: scenario.GazeboSimulator,
                             plot_facing_directions: bool = True,
                             plot_base_velocities: bool = False) -> None:
    """Visualize the retargeted frames along with the associated local features."""

    window_length_frames = local_window_features.window_length_frames
    window_step = local_window_features.window_step
    window_indexes = local_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    for i in range(initial_frame, final_frame):

        # Debug
        print(i - initial_frame, "/", final_frame - initial_frame)

        # The ik solutions are stored at double frequency w.r.t. the extracted features
        ik_solution = ik_solutions[2 * i]

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_position = np.asarray(ik_solution["base_position"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Retrieve local features
        base_positions = local_window_features.base_positions[i - window_length_frames]
        facing_directions = local_window_features.facing_directions[i - window_length_frames]
        base_velocities = local_window_features.base_velocities[i - window_length_frames]

        # =================
        # FACING DIRECTIONS
        # =================

        if plot_facing_directions:

            # Figure 1 for the facing directions
            plt.figure(1)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(-base_position[1], base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(-base_position[1], base_position[0], c='k')

                # Plot facing directions
                facing_direction = facing_directions[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current facing direction in blue
                    plt.plot([-base_position[1], -base_position[1] - 2 * facing_direction[1]],
                             [base_position[0], base_position[0] + 2 * facing_direction[0]], 'b',
                             label="Current facing direction")
                else:
                    # Other facing directions in green
                    plt.plot([-base_position[1], -base_position[1] - facing_direction[1]],
                             [base_position[0], base_position[0] + facing_direction[0]], 'g')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-0.75, 0.75])
            plt.ylim([-0.75, 0.75])
            plt.title("Facing directions (local view)")
            plt.legend()

        # ===============
        # BASE VELOCITIES
        # ===============

        if plot_base_velocities:

            # Figure 2 for the base velocities
            plt.figure(2)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(-base_position[1], base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(-base_position[1], base_position[0], c='k')

                # Plot base velocities
                base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current base velocity in magenta
                    plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                             [base_position[0], base_position[0] + base_velocity[0]], 'm',
                             label="Current base velocity")
                else:
                    # Other base velocities in gray
                    plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                             [base_position[0], base_position[0] + base_velocity[0]], 'gray')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-0.75, 0.75])
            plt.ylim([-0.75, 0.75])
            plt.title("Base velocities (local view)")
            plt.legend()

        # Plot
        plt.show()
        plt.pause(0.0001)
