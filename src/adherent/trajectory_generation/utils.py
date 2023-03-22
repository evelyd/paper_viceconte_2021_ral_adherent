# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
from scenario import core
from typing import List, Dict
from dataclasses import dataclass
from gym_ignition.utils import misc
from scenario import gazebo as scenario
from adherent.MANN.utils import read_from_file
from adherent.data_processing.utils import iCub

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =====================
# MODEL INSERTION UTILS
# =====================

@dataclass
class SphereURDF:
    """Class for defining a sphere urdf with parametric radius and color."""

    radius: float = 0.5
    color: tuple = (1, 1, 1, 1)

    def urdf(self) -> str:
        i = 2.0 / 5 * 1.0 * self.radius * self.radius
        urdf = f"""
            <robot name="sphere_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

                <!-- ====== -->
                <!-- COLORS -->
                <!-- ====== -->
                <material name="custom">
                    <color rgba="{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}"/>
                </material>
                <gazebo reference="sphere">
                    <visual>
                      <material>
                        <diffuse>{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}</diffuse>
                      </material>
                    </visual>
                    <collision>
                        <surface>
                          <friction>
                            <ode>
                              <mu>0.0</mu>
                            </ode>
                          </friction>
                        </surface>
                    </collision>
                </gazebo>

                <!-- ===== -->
                <!-- LINKS -->
                <!-- ===== -->

                <link name="sphere">
                    <inertial>
                      <origin rpy="0 0 0" xyz="0 0 0"/>
                      <mass value="1.0"/>
                      <inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/>
                    </inertial>
                    <visual>
                      <geometry>
                        <sphere radius="{self.radius}"/>
                      </geometry>
                      <origin rpy="0 0 0" xyz="0 0 0"/>
                      <material name="custom">
                        <color rgba="{self.color[0]} {self.color[1]} {self.color[2]} {self.color[3]}"/>
                      </material>
                    </visual>
                    <collision>
                      <geometry>
                        <sphere radius="{self.radius}"/>
                      </geometry>
                      <origin rpy="0 0 0" xyz="0 0 0"/>
                    </collision>
                </link>
                <gazebo reference="sphere">
                  <collision>
                    <surface>
                      <friction>
                        <ode>
                          <mu>0.0</mu>
                          <mu2>0.0</mu2>
                        </ode>
                      </friction>
                    </surface>
                  </collision>
                </gazebo>
            </robot>"""

        return urdf

class Shape:
    """Helper class to simplify shape insertion."""

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 model_string: str = SphereURDF(radius=0.02).urdf()):
        self.sdf = misc.string_to_file(model_string)

        # Assing incremental default name when multiple shapes are inserted
        name = scenario.get_model_name_from_sdf(self.sdf)
        index = 0
        while name in world.model_names():
            name = f"{name}{index}"

        # Insert the shape in the world
        assert world.insert_model(self.sdf, core.Pose(position, orientation), name)

        # Get and store the model and the world
        self.model = world.get_model(model_name=name)
        self.world = world

# =====================
# JOYSTICK DEVICE UTILS
# =====================

def quadratic_bezier(p0: np.array, p1: np.array, p2: np.array, t: np.array) -> List:
    """Define a discrete quadratic Bezier curve. Given the initial point p0, the control point p1 and
       the final point p2, the quadratic Bezier consists of t points and is defined by:
               Bezier(p0, p1, p2, t) = (1 - t)^2 p0 + 2t (1 - t) p1 + t^2 p2
    """

    quadratic_bezier = []

    for t_i in t:
        p_i = (1 - t_i) * (1 - t_i) * p0 + 2 * t_i * (1 - t_i) * p1 + t_i * t_i * p2
        quadratic_bezier.append(p_i)

    return quadratic_bezier

def compute_angle_wrt_x_positive_semiaxis(current_base_direction: List) -> float:
    """Compute the angle between the current base direction and the x positive semiaxis."""

    # Define the x positive semiaxis
    x_positive_semiaxis = np.asarray([1, 0])

    # Compute the yaw between the current base direction and the world x axis
    cos_theta = np.dot(x_positive_semiaxis, current_base_direction) # unitary norm vectors
    sin_theta = np.cross(x_positive_semiaxis, current_base_direction) # unitary norm vectors
    angle = math.atan2(sin_theta, cos_theta)

    return angle

# ===========================
# TRAJECTORY GENERATION UTILS
# ===========================

def define_initial_nn_X(robot: str) -> List:
    """Define the robot-specific initial input X for the network used for trajectory generation."""

    if robot != "iCubV2_5":
        raise Exception("Initial network input X only defined for iCubV2_5.")

    # Initial input manually retrieved from a standing pose
    initial_nn_X = [[3.5243694266353373e-15, 7.008199078189101e-15, 3.0614383132617574e-16, 0.012421162571032856,
                     -0.014122548398467332, -0.0017588395790161734, 0.0044019864826900425, -0.0032526431671412843,
                     -0.0003791446290885935, 0.0018464051771702283, -0.0007014617207248446, -0.0002852708574057455,
                     -0.007666416570001765, -0.006401494346322381, -0.0020622161550935647, -0.0019045289643069853,
                     -0.008158728689814406, 0.0006715067872605493, 0.0007060098803055911, -0.01477829078637679, 
                     0.0011823591653468393, -0.0003210022438017331, 6.757567991447852e-06, 0.003540728250141397, 
                     0.0, 0.0, 0.0, 0.017893397271105622, -0.03658355242181092, 0.006983540281113693,
                     0.09451721426339721, -0.009889970586401762, 0.008193869415440216, 0.2053173727694591, 
                     0.07173449613753384, 0.009762607489653094, 1.2839528469302477e-16, 1.9898256768176927e-17, 
                     6.762609578082937e-18, 0.031449943566089156, 0.07414012486813136, 0.07208904404786341, 
                     0.006826734024380377, 0.050238738599215724, 0.0016967450914463823, 0.004713732379144265, 
                     0.019225556089882915, -0.008454823733790681, 0.019126096353924676, -0.03700523273782162, 
                     0.02385417785340806, 0.007936443095453556, -0.005759098538862559, 0.016742656763119815, 
                     0.00818785217095963, -0.037435052638151126, -0.016759287151117987, 0.05256457282727177, 
                     -0.03841732944090507, -0.10431413289643675, 0.0, 0.0, 0.0, 0.0878491471703012,
                     -0.12733981016516405, -0.2832203614502011, 0.03918120592226204, 0.0804221698146093,
                     -0.43534358315740246, -0.09585916148288552, 0.15195461008141126, -0.6545870342808059, 
                     -0.08914329577232137, 0.025767620112200747, 0.016600125582731447, -0.10205569019576242, 
                     -0.10115357046332556, -0.02590094449134414, -0.10954097755732813, -0.021888724926318617, 
                     0.06819316643211669, -0.07852679097651347, -0.10034170556770548, 0.020710812052444683, 
                     0.2413038455611485, 0.010535350226309968, 0.006275324178053386, -0.14908520025814864, 
                     0.08533421586871431, 0.10281322318047023, 0.32297257397277707, 0.14790247588361516, 
                     -0.3129451427487485, 0.04242320961879248, 0.0022412263648723028, -0.016729676987423055, 
                     0.11047971812597632, -0.12993538373842523, 0.018668904587540704, 0.033567343049341246, 
                     0.3631242921725555, -0.28302209132906536, -0.3129451427487485, 0.04242320961879248, 
                     0.018265725998252436, -0.0035976586400216642, -0.012043915220395253, -0.010850374465225243, 
                     -0.0029826472872050702, 0.01194233827184095, 0.00894458674770629, 0.003558151484858718, 
                     0.01579702497265592, -0.0028171693990643176, -0.001904114327552775, -0.011985210234407781, 
                     0.020805414325628102, 0.011839621139096701, 0.009719580928183296, 0.01040432230648286, 
                     0.03174781826678491, -0.07732321906598477, -0.051694399002594205, 0.015153131453655988, 
                     8.599305395817769e-05, -1.1664542210951256e-05, 0.005543033652851423, 0.0010366246463409945, 
                     -0.01100933606651422, -0.041149377276150645, -0.0024410871334947654, -0.04243541029015849, 
                     0.01042528371749396, 0.0009363820684421542, 8.599305395817769e-05, -1.1664542210951256e-05]]

    return initial_nn_X

def define_initial_past_trajectory(robot: str) -> (List, List, List):
    """Define the robot-specific initialization of the past trajectory data used for trajectory generation."""

    if robot != "iCubV2_5":
        raise Exception("Initial past trajectory data only defined for iCubV2_5.")

    # The above quantities are expressed in the frame specified by the initial base position and orientation

    # Initial past base linear velocities manually retrieved from a standing pose
    initial_past_trajectory_base_vel = [[-0.00716343,  0.00385182,  0.00026315],
                                        [ 3.83626058e-15,  7.09872475e-15, -5.22836963e-15], 
                                        [-0.00675958,  0.00449099,  0.00059273], 
                                        [-0.00722386,  0.00551062,  0.00051982], 
                                        [-7.03736213e-15,  3.44515696e-15, -4.85989740e-16], 
                                        [-0.00676486,  0.00158084, -0.00026224], 
                                        [-0.00538534,  0.00128945,  0.00017258], 
                                        [-0.00331833,  0.00163447,  0.00042496], 
                                        [-0.00953416,  0.00254862,  0.00044421], 
                                        [-1.80269867e-15,  5.24949718e-15, -8.90842611e-17], 
                                        [-0.01330885,  0.00049532,  0.00115734], 
                                        [4.83036766e-15, 1.75945287e-15, 5.93309633e-15], 
                                        [-0.00349938, -0.00056845,  0.00053368], 
                                        [-0.01246749, -0.00152131,  0.000421  ], 
                                        [-0.00832295,  0.0004091 , -0.00032585], 
                                        [-0.01797435,  0.00017675, -0.00029361], 
                                        [-0.01288769,  0.00301351,  0.00084927], 
                                        [-0.00877806, -0.00043779, -0.00075292], 
                                        [ 1.40747243e-14, -6.89031391e-15,  9.71979480e-16], 
                                        [-0.02340612,  0.00397878, -0.00091494], 
                                        [-6.63306633e-15,  3.49004431e-15, -6.02218059e-15], 
                                        [-0.00865152,  0.00134653, -0.00177219], 
                                        [-0.00786997, -0.00233357, -0.00156049], 
                                        [-0.00892424, -0.00258245, -0.00146873], 
                                        [-0.00811743,  0.00200219, -0.00099574], 
                                        [-0.01302307,  0.00418385, -0.00174521], 
                                        [-0.00462916,  0.00026048, -0.00119211], 
                                        [-0.0050965 , -0.00139854, -0.00160158], 
                                        [-0.005476  , -0.00227114, -0.00200646], 
                                        [-0.01676012, -0.00561986, -0.00489061], 
                                        [ 0.00264328, -0.00101225, -0.00052074], 
                                        [ 5.40809601e-15, -1.57484915e-14,  2.67252783e-16], 
                                        [-0.0027995 , -0.00375919, -0.00231512], 
                                        [ 1.80269867e-15, -5.24949718e-15,  8.90842611e-17], 
                                        [ 0.01081781, -0.00782894, -0.00248261], 
                                        [ 0.00139013, -0.00224276, -0.00093824], 
                                        [-0.00658927, -0.00711429, -0.00251821], 
                                        [-4.97053540e-16,  2.66963594e-15, -5.58073298e-15], 
                                        [ 0.00104704, -0.00982987, -0.00381658], 
                                        [ 0.00307704, -0.00356778, -0.00104486], 
                                        [-0.00066357, -0.00226427, -0.00036987], 
                                        [-0.01301264, -0.00357357, -0.00220798], 
                                        [-0.00417191, -0.00309992, -0.00233419], 
                                        [ 0.00640315, -0.00554543, -0.00296235], 
                                        [ 0.00687916, -0.00118332, -0.00065244], 
                                        [ 0.00819684, -0.00063089, -0.00048392], 
                                        [ 0.00726917, -0.00193537, -0.00091059], 
                                        [ 0.00651411, -0.00352663, -0.00144585], 
                                        [ 0.02139752, -0.00941589, -0.00282025], 
                                        [-3.51263960e-15,  7.78447106e-15,  1.09387553e-14], 
                                        [ 0.00912293, -0.00267039, -0.00048275]]

    # Initial past base angular velocities manually retrieved from a standing pose
    initial_past_trajectory_base_ang_vel = [[-0.00755611, -0.0350876 , -0.0231277 ], 
                                            [0., 0., 0.], [-0.0087613 , -0.03888634, -0.027938  ], 
                                            [-0.0105522 , -0.04917967, -0.00152774], 
                                            [ 2.84504292e-16, -6.17399465e-16,  2.75187740e-15], 
                                            [-0.00168213, -0.04111259, -0.00753031], 
                                            [-0.00309745, -0.03786141, -0.03232349], 
                                            [-0.00704214, -0.02328724, -0.00895122], 
                                            [-0.01365813, -0.05655421,  0.00088823], 
                                            [8.54022144e-17, 1.48861722e-17, 2.82559492e-18], 
                                            [-0.0308838 , -0.07690305,  0.00853956], 
                                            [-1.70797352e-16, -2.97883444e-17, -5.77338249e-18], 
                                            [-0.00905094, -0.02452463, -0.00801273], 
                                            [-0.01191892, -0.07322378, -0.02738323], 
                                            [-0.00200355, -0.0343647 , -0.02822263], 
                                            [-0.00230115, -0.06715387, -0.11565876], 
                                            [-0.01534877, -0.05918849, -0.10457992], 
                                            [ 0.00594433, -0.04117897, -0.04580102], 
                                            [ 2.42486990e-16, -3.43752578e-16, -7.26415095e-15], 
                                            [ 0.00069116, -0.08983208, -0.1398635 ], 
                                            [-7.18915658e-16,  2.69132944e-15, -3.23792190e-17], 
                                            [ 0.01061373, -0.02487456, -0.00066224], 
                                            [ 0.01487274, -0.03206658, -0.02535907], 
                                            [ 0.01293469, -0.03481197, -0.01976739], 
                                            [ 0.00374153, -0.02509881, -0.0024132 ], 
                                            [ 0.00458413, -0.01476757,  0.01782594], 
                                            [ 0.00892151, -0.00587989,  0.01157207], 
                                            [ 0.01427018, -0.01349857,  0.01140695], 
                                            [ 0.02126881, -0.01271533, -0.00112174], 
                                            [ 0.04419007, -0.02972633, -0.01915061], 
                                            [0.00704018, 0.0167406 , 0.01190216], 
                                            [8.54525993e-17, 1.44741001e-17, 3.38442587e-18], 
                                            [ 0.02465543,  0.0231271 , -0.0269139 ], 
                                            [ 3.08885256e-16, -1.71101691e-17, -5.46882732e-15], 
                                            [0.03967082, 0.0946244 , 0.00762483], 
                                            [ 0.01492882,  0.01411408, -0.01669203], 
                                            [ 0.03444098, -0.027689  , -0.05422716], 
                                            [ 1.45933105e-16,  6.63493579e-16, -5.47416306e-15], 
                                            [ 0.05454233,  0.06117371, -0.06128212], 
                                            [ 0.01988478,  0.02682925, -0.02975806], 
                                            [ 0.00822804, -0.01885205, -0.03093768], 
                                            [ 0.0309434 , -0.07454498, -0.11478169], 
                                            [ 0.03355912, -0.00444326, -0.06843284], 
                                            [ 0.0540644 ,  0.03668021, -0.05470951], 
                                            [0.01472837, 0.02494858, 0.01199966], 
                                            [0.01405378, 0.03205797, 0.02029692], 
                                            [0.01825257, 0.03231723, 0.01303322], 
                                            [0.02497076, 0.02872931, 0.00766891], 
                                            [0.05355659, 0.04849043, 0.13106968], 
                                            [1.28251216e-16, 2.13485038e-17, 4.78841635e-18], 
                                            [0.01283283, 0.02187821, 0.07408406]]

    return initial_past_trajectory_base_vel, initial_past_trajectory_base_ang_vel

def define_initial_base_height(robot: str) -> List:
    """Define the robot-specific initial height of the base frame."""

    if robot != "iCubV2_5":
        raise Exception("Initial base height only defined for iCubV2_5.")

    initial_base_height = 0.6354

    return initial_base_height

def define_initial_base_angle(robot: str) -> List:
    """Define the robot-specific initial base yaw expressed in the world frame."""

    if robot != "iCubV2_5":
        raise Exception("Initial base yaw only defined for iCubV2_5.")

    # For iCubV2_5, the initial base yaw is 180 degs since the x axis of the base frame points backward
    initial_base_angle = np.array([0.0, 0.0, math.pi])

    return initial_base_angle

def trajectory_blending(a0: List, a1: List, t: np.array, tau: float) -> List:
    """Blend the vectors a0 and a1 via:
           Blend(a0, a1, t, tau) = (1 - t^tau) a0 + t^tau a1
       Increasing tau means biasing more towards a1.
    """

    blended_trajectory = []

    for i in range(len(t)):
        p_i = (1 - math.pow(t[i], tau)) * np.array(a0[i]) + math.pow(t[i], tau) * np.array(a1[i])
        blended_trajectory.append(p_i.tolist())

    return blended_trajectory

def load_component_wise_input_mean_and_std(datapath: str) -> (Dict, Dict):
    """Compute component-wise input mean and standard deviation."""

    # Full-input mean and std
    Xmean = read_from_file(datapath + 'X_mean.txt')
    Xstd = read_from_file(datapath + 'X_std.txt')

    # Remove zeroes from Xstd
    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1

    # Retrieve component-wise input mean and std (used to normalize the next input for the network)
    Xmean_dict = {"past_base_velocities": Xmean[0:18]}
    Xstd_dict = {"past_base_velocities": Xstd[0:18]}
    Xmean_dict["future_base_velocities"] = Xmean[18:36]
    Xstd_dict["future_base_velocities"] = Xstd[18:36]
    Xmean_dict["past_base_angular_velocities"] = Xmean[36:54]
    Xstd_dict["past_base_angular_velocities"] = Xstd[36:54]
    Xmean_dict["future_base_angular_velocities"] = Xmean[54:72]
    Xstd_dict["future_base_angular_velocities"] = Xstd[54:72]
    Xmean_dict["s"] = Xmean[72:104]
    Xstd_dict["s"] = Xstd[72:104]
    Xmean_dict["s_dot"] = Xmean[104:]
    Xstd_dict["s_dot"] = Xstd[104:]

    return Xmean_dict, Xstd_dict

def load_output_mean_and_std(datapath: str) -> (List, List):
    """Compute output mean and standard deviation."""

    # Full-output mean and std
    Ymean = read_from_file(datapath + 'Y_mean.txt')
    Ystd = read_from_file(datapath + 'Y_std.txt')

    # Remove zeroes from Ystd
    for i in range(Ystd.size):
        if Ystd[i] == 0:
            Ystd[i] = 1

    return Ymean, Ystd

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_generated_motion(icub: iCub,
                               gazebo: scenario.GazeboSimulator,
                               posturals: Dict,
                               raw_data: List,
                               blending_coeffs: Dict,
                               plot_blending_coeffs: bool) -> None:
    """Visualize the generated motion along with the joystick inputs used to generate it and,
    optionally, the activations of the blending coefficients during the trajectory generation."""

    # Retrieve joint and base posturals
    joint_posturals = posturals["joints"]
    base_posturals = posturals["base"]

    # Retrieve blending coefficients
    if plot_blending_coeffs:
        w_1 = blending_coeffs["w_1"]
        w_2 = blending_coeffs["w_2"]
        w_3 = blending_coeffs["w_3"]
        w_4 = blending_coeffs["w_4"]

    # Define controlled joints
    controlled_joints = icub.joint_names()

    # Plot configuration
    plt.ion()

    for frame_idx in range(len(joint_posturals)):

        # Debug
        print(frame_idx, "/", len(joint_posturals))

        # ======================
        # VISUALIZE ROBOT MOTION
        # ======================

        # Retrieve the current joint positions
        joint_postural = joint_posturals[frame_idx]
        joint_positions = [joint_postural[joint] for joint in controlled_joints]

        # Retrieve the current base position and orientation
        base_postural = base_posturals[frame_idx]
        base_position = base_postural['postion']
        base_quaternion = base_postural['wxyz_quaternions']

        # Reset the robot configuration in the simulator
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # =====================================
        # PLOT THE MOTION DIRECTION ON FIGURE 1
        # =====================================

        # Retrieve the current motion direction
        curr_raw_data = raw_data[frame_idx]
        curr_x = curr_raw_data[0]
        curr_y = curr_raw_data[1]

        plt.figure(1)
        plt.clf()

        # Circumference of unitary radius
        r = 1
        x = np.linspace(-r, r, 1000)
        y = np.sqrt(-x ** 2 + r ** 2)
        plt.plot(x, y, 'r')
        plt.plot(x, -y, 'r')

        # Motion direction
        plt.scatter(0, 0, c='r')
        desired_motion_direction = plt.arrow(0, 0, curr_x, -curr_y, length_includes_head=True, width=0.01,
                                             head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='r')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.4, 1.2])
        plt.axis('off')
        plt.legend([desired_motion_direction], ['DESIRED MOTION DIRECTION'], loc="lower center")

        # =====================================
        # PLOT THE FACING DIRECTION ON FIGURE 2
        # =====================================

        # Retrieve the current facing direction
        curr_z = curr_raw_data[2]
        curr_rz = curr_raw_data[3]

        plt.figure(2)
        plt.clf()

        # Circumference of unitary norm
        r = 1
        x = np.linspace(-r, r, 1000)
        y = np.sqrt(-x ** 2 + r ** 2)
        plt.plot(x, y, 'b')
        plt.plot(x, -y, 'b')

        # Facing direction
        plt.scatter(0, 0, c='b')
        desired_facing_direction = plt.arrow(0, 0, curr_z, -curr_rz, length_includes_head=True, width=0.01,
                                             head_width=8 * 0.01, head_length=1.8 * 8 * 0.01, color='b')

        # Plot configuration
        plt.axis('scaled')
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.4, 1.2])
        plt.axis('off')
        plt.legend([desired_facing_direction], ['DESIRED FACING DIRECTION'], loc="lower center")

        # ==========================================
        # PLOT THE BLENDING COEFFICIENTS ON FIGURE 3
        # ==========================================

        if plot_blending_coeffs:
            # Retrieve the blending coefficients up to the current time
            curr_w_1 = w_1[:frame_idx]
            curr_w_2 = w_2[:frame_idx]
            curr_w_3 = w_3[:frame_idx]
            curr_w_4 = w_4[:frame_idx]

            plt.figure(3)
            plt.clf()

            plt.plot(range(len(curr_w_1)), curr_w_1, 'r')
            plt.plot(range(len(curr_w_2)), curr_w_2, 'b')
            plt.plot(range(len(curr_w_3)), curr_w_3, 'g')
            plt.plot(range(len(curr_w_4)), curr_w_4, 'y')

            # Plot configuration
            plt.title("Blending coefficients profiles")
            plt.xlim([0, len(w_1)])
            plt.ylabel("Blending coefficients")
            plt.xlabel("Time [s]")

        # Plot
        plt.show()
        plt.pause(0.0001)

    input("Press Enter to end the visualization of the generated trajectory.")
