# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

# TODO: manually set the env variable at each restart of the docker container
# export IGN_FILE_PATH=/home/pviceconte/git/adherent_ergocub/src/adherent/model/ergoCubGazeboV1_xsens/:$IGN_FILE_PATH

# Use tf version 2.3.0 as 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import yarp
import time
import argparse
import numpy as np
from scenario import gazebo as scenario
from adherent.data_processing.utils import iCub
from gym_ignition.utils.scenario import init_gazebo_sim
from gym_ignition.rbd.idyntree import kindyncomputations
from adherent.data_processing.utils import define_foot_vertices
from adherent.trajectory_generation import trajectory_generator
from adherent.trajectory_generation.utils import define_initial_nn_X
from adherent.data_processing.utils import define_feet_frames_and_links
from adherent.trajectory_generation.utils import define_base_pitch_offset
from adherent.trajectory_generation.utils import define_initial_support_foot_and_vertex
from adherent.trajectory_generation.utils import define_initial_base_angle
from adherent.trajectory_generation.utils import define_initial_base_height
from adherent.trajectory_generation.utils import define_initial_past_trajectory

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--storage_path", help="Path where the generated trajectory will be stored. Relative path from script folder.",
                    type=str, default="../datasets/inference/")
parser.add_argument("--training_path", help="Path where the training-related data are stored. Relative path from script folder.",
                    # type=str, default="../datasets/training_subsampled_mirrored_D2_D3_20230319-112816/")
                    type = str, default = "../datasets/training_subsampled_mirrored_D2_D3_20230320-190551/")

parser.add_argument("--save_every_N_iterations", help="Data will be saved every N iterations.",
                    type=int, default=1000)
parser.add_argument("--plot_trajectory_blending", help="Visualize the blending of the future ground trajectory to build the next network input.", action="store_true")
parser.add_argument("--plot_footsteps", help="Visualize the footsteps.", action="store_true")
parser.add_argument("--plot_blending_coefficients", help="Visualize blending coefficient activations.", action="store_true")
parser.add_argument("--time_scaling", help="Time scaling to be applied to the generated trajectory. Keep it integer.",
                    type=int, default=1)
parser.add_argument("--plot_contacts", help="Visualize the contacts extracted for the controller.", action="store_true")

args = parser.parse_args()

storage_path = args.storage_path
training_path = args.training_path
save_every_N_iterations = args.save_every_N_iterations
plot_trajectory_blending = args.plot_trajectory_blending
plot_footsteps = args.plot_footsteps
plot_blending_coefficients = args.plot_blending_coefficients
time_scaling = args.time_scaling
plot_contacts = args.plot_contacts

# ==================
# YARP CONFIGURATION
# ==================

# Inter-process communication is implemented via YARP ports, therefore "yarp server"
# needs to be executed in a separate terminal before launching this script

# Initialise YARP
yarp.Network.init()

# Open and connect YARP port to retrieve joystick input
p_in = yarp.BufferedPortBottle()
p_in.open("/joystick_in")
yarp.Network.connect("/joystick_out", p_in.getName())

# Initialization of the joystick raw and processed inputs received through YARP ports
raw_data = []
base_velocities = []
base_angular_velocities = []

# ===============
# MODEL INSERTION
# ===============

# Set scenario verbosity
scenario.set_verbosity(scenario.Verbosity_warning)

# Get the default simulator and the default empty world
gazebo, world = init_gazebo_sim()

# Retrieve the robot urdf model
script_directory = os.path.dirname(os.path.abspath(__file__))
icub_urdf = os.path.join(script_directory, "../src/adherent/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

# Insert the robot in the empty world
initial_base_height = define_initial_base_height(robot="ergoCubV1")
icub = iCub(world=world, urdf=icub_urdf, position=[0, 0, initial_base_height])

# Show the GUI
gazebo.gui()
gazebo.run(paused=True)

# Get the robot joints
icub_joints = icub.joint_names()

# Define the joints of interest for the features computation and their associated indexes in the robot joints  list
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'l_wrist_yaw', 'l_wrist_roll', 'l_wrist_pitch', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'r_wrist_yaw', 'r_wrist_roll', 'r_wrist_pitch'] # right arm
controlled_joints_indexes = [icub_joints.index(elem) for elem in controlled_joints]
# controlled_joints_retrieved = [icub_joints[index] for index in controlled_joints_indexes] # This is how to use the indexes


# Create a KinDynComputations object
kindyn = kindyncomputations.KinDynComputations(model_file=icub_urdf, considered_joints=controlled_joints)
kindyn.set_robot_state_from_model(model=icub, world_gravity=np.array(world.gravity()))

# ==================
# PLOT CONFIGURATION
# ==================

# Define the indexes of the figures for plotting
figure_base_vel = 1
figure_base_ang_vel = 2
figure_blending_coefficients = 3
figure_footsteps = 4
plt.ion()

# ====================
# TRAJECTORY GENERATOR
# ====================

# Trajectory control and trajectory generation rates
generation_rate = 1/50
control_rate = 1/100

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = define_foot_vertices(robot="ergoCubV1")

# Define robot-specific initial input X for the trajectory generation neural network
initial_nn_X = define_initial_nn_X(robot="ergoCubV1")

# Define robot-specific initial past trajectory features
initial_past_trajectory_base_vel, initial_past_trajectory_base_ang_vel = \
    define_initial_past_trajectory(robot="ergoCubV1")

# Define robot-specific initial base yaw angle
initial_base_angle = define_initial_base_angle(robot="ergoCubV1")

# Define robot-specific feet frames and links
feet_frames, feet_links = define_feet_frames_and_links(robot="ergoCubV1")

# Define robot-specific initial support foot and vertex
initial_support_foot, initial_support_vertex = define_initial_support_foot_and_vertex(robot="ergoCubV1")

# Define robot-specific (and possibly training-dependent) base pitch offset
base_pitch_offset = define_base_pitch_offset(robot="ergoCubV1")

# Instantiate the trajectory generator
generator = trajectory_generator.TrajectoryGenerator.build(icub=icub, gazebo=gazebo, kindyn=kindyn,
                                                           controlled_joints_indexes=controlled_joints_indexes,
                                                           storage_path=os.path.join(script_directory, storage_path),
                                                           training_path=os.path.join(script_directory, training_path),
                                                           local_foot_vertices_pos=local_foot_vertices_pos,
                                                           feet_frames=feet_frames,
                                                           feet_links=feet_links,
                                                           initial_nn_X=initial_nn_X,
                                                           initial_past_trajectory_base_vel=initial_past_trajectory_base_vel,
                                                           initial_past_trajectory_base_ang_vel=initial_past_trajectory_base_ang_vel,
                                                           initial_base_height=initial_base_height,
                                                           initial_base_angle=initial_base_angle,
                                                           initial_support_foot=initial_support_foot,
                                                           initial_support_vertex=initial_support_vertex,
                                                           time_scaling=time_scaling,
                                                           generation_rate=generation_rate,
                                                           control_rate=control_rate)

# =========
# MAIN LOOP
# =========

while True:

    # Update the iteration counter
    generator.update_iteration_counter()

    # Retrieve the network output
    current_output, denormalized_current_output = generator.retrieve_network_output_pytorch()

    # TODO: add the blending coefficients (it should be something like the following)
    # # Retrieve the network output and the blending coefficients
    # current_output, denormalized_current_output, current_blending_coefficients = \
    #     generator.retrieve_network_output_and_blending_coefficients(nn_X=nn_X,
    #                                                                 nn_keep_prob=nn_keep_prob,
    #                                                                 output=output,
    #                                                                 blending_coefficients=blending_coefficients)

    # Apply the joint positions and the base orientation from the network output
    joint_positions, joint_velocities, new_base_quaternion = generator.apply_joint_positions_and_base_orientation(
        denormalized_current_output=denormalized_current_output,
        base_pitch_offset=base_pitch_offset)

    # Handle first iteration differently
    if generator.iteration > 1:

        # Update the support vertex position
        generator.update_support_vertex_position()

        # Compute kinematically-feasible base position and updated posturals
        new_base_postural, new_joints_pos_postural, new_joints_vel_postural, new_links_postural, \
        new_com_pos_postural, new_com_vel_postural, new_centroidal_momentum_postural = \
            generator.compute_kinematically_feasible_base_and_update_posturals(joint_positions=joint_positions,
                                                                            joint_velocities=joint_velocities,
                                                                            base_quaternion=new_base_quaternion,
                                                                            controlled_joints=controlled_joints,
                                                                            link_names=icub.link_names())

    # Update the support foot and vertex while detecting new footsteps
    support_foot, update_footsteps_list = generator.update_support_vertex_and_support_foot_and_footsteps()

    if update_footsteps_list and plot_footsteps:

        # Plot the last footstep
        generator.plotter.plot_new_footstep(figure_footsteps=figure_footsteps,
                                            support_foot=support_foot,
                                            new_footstep=generator.storage.footsteps[support_foot][-1])

    # Compute kinematically-feasible base position and updated posturals
    new_base_postural, new_joints_pos_postural, new_joints_vel_postural, new_links_postural, \
    new_com_pos_postural, new_com_vel_postural, new_centroidal_momentum_postural = \
        generator.compute_kinematically_feasible_base_and_update_posturals(joint_positions=joint_positions,
                                                                            joint_velocities=joint_velocities,
                                                                            base_quaternion=new_base_quaternion,
                                                                            controlled_joints=controlled_joints,
                                                                            link_names=icub.link_names())

    # Retrieve user input data from YARP port
    base_velocities, base_angular_velocities, raw_data = \
        generator.retrieve_joystick_inputs(input_port=p_in,
                                            base_velocities=base_velocities,
                                            base_angular_velocities=base_angular_velocities,
                                            raw_data=raw_data)

    # Use in an autoregressive fashion the network output and blend it with the user input
    blended_base_velocities, blended_base_angular_velocities = \
        generator.autoregression_and_blending(current_output=current_output,
                                                denormalized_current_output=denormalized_current_output,
                                                base_velocities=base_velocities,
                                                base_angular_velocities=base_angular_velocities)

    # Update storage and periodically save data
    generator.update_storages_and_save(blending_coefficients=[1,1,1,1], # TODO: temporary fix to handle blending coefficients
                                        base_postural=new_base_postural,
                                        joints_pos_postural=new_joints_pos_postural,
                                        joint_vel_postural=new_joints_vel_postural,
                                        links_postural=new_links_postural,
                                        com_pos_postural=new_com_pos_postural,
                                        com_vel_postural=new_com_vel_postural,
                                        centroidal_momentum_postural=new_centroidal_momentum_postural,
                                        raw_data=raw_data,
                                        base_velocities=base_velocities,
                                        base_angular_velocities=base_angular_velocities,
                                        save_every_N_iterations=save_every_N_iterations,
                                        plot_contacts=plot_contacts)

    # TODO: temporary deactivation of the blending coefficients plots
    # if plot_trajectory_blending:
    #
    #     # Plot the trajectory blending
    #     generator.plotter.plot_trajectory_blending(figure_facing_dirs=figure_facing_dirs,
    #                                                figure_base_vel=figure_base_vel,
    #                                                denormalized_current_output=denormalized_current_output,
    #                                                quad_bezier=quad_bezier, facing_dirs=facing_dirs,
    #                                                base_velocities=base_velocities,
    #                                                blended_base_positions=blended_base_positions,
    #                                                blended_facing_dirs=blended_facing_dirs,
    #                                                blended_base_velocities=blended_base_velocities)
    #
    # if plot_blending_coefficients:
    #
    #     # Plot the blending coefficients
    #     generator.plotter.plot_blending_coefficients(figure_blending_coefficients=figure_blending_coefficients,
    #                                                  blending_coeffs=generator.storage.blending_coeffs)
    #
    # if plot_trajectory_blending or plot_footsteps or plot_blending_coefficients:
    #
    #     # Plot
    #     plt.show()
    #     plt.pause(0.0001)
    #
    # else:
    #
    #     # Slow down visualization
    #     time.sleep(0.0001)

    # Slow down visualization
    time.sleep(0.001)
