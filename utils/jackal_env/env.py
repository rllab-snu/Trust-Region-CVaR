from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

from scipy.spatial.transform import Rotation
from copy import deepcopy
from gym import spaces
import numpy as np
import time
import gym
# import cv2
import io
import os

def theta2vec(theta):
    ''' Convert an angle (in radians) to a unit vector in that angle around Z '''
    return np.array([np.cos(theta), np.sin(theta), 0.0])


class Env(gym.Env):
    def __init__(self):
        abs_path = os.path.dirname(__file__)
        self.model = load_model_from_path(f'{abs_path}/jackal.xml')
        self.time_step = 0.002
        self.n_substeps = 1
        self.time_step *= self.n_substeps
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        self.viewer = None

        # for environment
        self.pre_goal_dist = 0.0
        self.control_freq = 30
        self.num_time_step = int(1.0/(self.time_step*self.control_freq))
        self.limit_distance = 0.5
        self.limit_bound = 0.0
        self.hazard_size = 0.25*np.sqrt(2.0)
        self.goal_dist_threshold = 0.25
        self.h_coeff = 10.0
        self.max_steps = 1000
        self.cur_step = 0
        self.num_hazard = 8
        self.num_goal = 1
        self.num_candi_goal = 5
        self.hazard_group = 2
        self.num_group = 6

        # for candi pos list
        x_space = np.linspace(-2.25, 2.25, 10)
        y_space = np.linspace(-2.25, 2.25, 10)
        self.candi_pos_list = []
        self.candi_pos_indices = []
        cnt = 0
        for x_pos in x_space:
            for y_pos in y_space:
                if abs(x_pos) < 1.0 and abs(y_pos) < 1.0:
                    continue
                self.candi_pos_list.append([x_pos, y_pos])
                self.candi_pos_indices.append(cnt)
                cnt += 1

        # for PID control
        self.p_coeff = 10.0
        self.d_coeff = 0.001
        self.ang_p_coeff = 2.0
        self.ang_d_coeff = 0.001

        # for state
        self.angle_interval = 2
        self.angle_range = np.arange(-135.0, 135.0 + self.angle_interval, self.angle_interval)
        self.max_scan_value = 3.0
        self.max_goal_dist = 3.0
        self.scan_value = np.zeros(26, dtype=np.float32)
        self.robot_pose = np.zeros(3)
        self.robot_vel = np.zeros(2)
        self.pre_robot_vel = np.zeros(2)
        self.candi_goal_pos_list = []
        self.goal_pos = np.zeros(2)

        # for action
        self.action = np.zeros(2)

        # state & action dimension
        self.action_dim = 2
        self.state_dim = len(self.scan_value) + len(self.robot_pose) + len(self.robot_vel)
        self.state_dim += 1
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf*np.ones(self.state_dim), np.inf*np.ones(self.state_dim), dtype=np.float32)

        # hazard & goal idx list
        self.hazard_goal_idx_list = []
        self.hazard_goal_idx_list.append([13,14,15,16,83,84,85,86, 90, 99, 0, 9])
        self.hazard_goal_idx_list.append([21,22,27,28,81,82,87,88, 57, 51, 4])
        self.hazard_goal_idx_list.append([41,51,49,59,5,15,85,95, 90, 0, 9, 99])


    def getCost(self, h_dist):
        limit_d = self.limit_distance + self.limit_bound
        cost = 1.0/(1.0 + np.exp((h_dist - limit_d)*self.h_coeff))
        return cost

    def render(self, mode, **kwargs):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
        self.viewer.render()

    def getSensor(self):
        sensor_dict = {'accelerometer':None, 'velocimeter':None, 'gyro':None}
        for sensor_name in sensor_dict.keys():
            id = self.sim.model.sensor_name2id(sensor_name)
            adr = self.sim.model.sensor_adr[id]
            dim = self.sim.model.sensor_dim[id]
            sensor_dict[sensor_name] = self.sim.data.sensordata[adr:adr + dim].copy()
        return sensor_dict

    def getLidar(self):
        lidar_value = np.zeros_like(self.angle_range, dtype=np.float32)
        pos = self.sim.data.get_body_xpos('robot').copy()
        rot_mat = self.sim.data.get_body_xmat('robot').copy()
        body = self.sim.model.body_name2id('robot')
        grp = np.array([i==self.hazard_group for i in range(self.num_group)], dtype='uint8')
        for i, angle in enumerate(self.angle_range):
            rad_angle = angle*np.pi/180.0
            vec = np.matmul(rot_mat, theta2vec(rad_angle))
            dist, _ = self.sim.ray_fast_group(pos, vec, grp, 1, body)
            if dist > 0:
                lidar_value[i] = dist
            else:
                lidar_value[i] = self.max_scan_value
        for i in range(len(self.scan_value)):
            self.scan_value[i] = np.mean(lidar_value[5*i:5*i+11])
        return deepcopy(self.scan_value)

    # def _showLidarImage(self, lidar, default_ang=130.0, size=128):
    #     img = np.zeros((2*size, 2*size))
    #     for i in range(len(lidar)):
    #         dist = int((1.0 - lidar[i])*size)
    #         start_angle = i*10.0 - default_ang
    #         end_angle = (i+1)*10.0 - default_ang
    #         img = cv2.ellipse(img, (size, size), (dist,dist), 0, -start_angle, -end_angle, 1.0, -1)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(1)
    #     return img
    
    def getState(self):
        self.sim.forward()
        sensor_dict = self.getSensor()
        self.robot_vel[0] = sensor_dict['velocimeter'][0]
        self.robot_vel[1] = sensor_dict['gyro'][2]
        robot_acc = np.array([self.robot_vel[0] - self.pre_robot_vel[0]])*self.control_freq
        self.pre_robot_vel = deepcopy(self.robot_vel)

        self.robot_pose = self.sim.data.get_body_xpos('robot').copy()
        robot_mat = self.sim.data.get_body_xmat('robot').copy()
        theta = Rotation.from_matrix(robot_mat).as_euler('zyx', degrees=False)[0]
        self.robot_pose[2] = theta

        rel_goal_pos = self.goal_pos - self.robot_pose[:2]
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        rel_goal_pos = np.matmul(rot_mat, rel_goal_pos)
        goal_dist = np.linalg.norm(rel_goal_pos)
        goal_dir = rel_goal_pos/(goal_dist + 1e-8)

        vel = deepcopy(self.robot_vel)
        scan_value = self.getLidar()
        state = {'goal_dir':goal_dir,
                'goal_dist':goal_dist,
                'vel':vel,
                'acc':robot_acc,
                'scan':scan_value}
        return state

    def getFlattenState(self, state):
        goal_dir = state['goal_dir']
        goal_dist = [np.clip(state['goal_dist'], 0.0, self.max_goal_dist)]
        vel = state['vel']
        acc = state['acc']
        scan = 1.0 - (np.clip(state['scan'], 0.0, self.max_scan_value)/self.max_scan_value)
        state = np.concatenate([goal_dir, goal_dist, vel, acc/8.0, scan], axis=0)
        return state

    def build(self):
        self.sim.reset()

        while True:
            sampled_candi_indices = np.random.choice(self.candi_pos_indices, self.num_hazard + self.num_candi_goal, replace=False)
            hazard_pos_list = [np.array(self.candi_pos_list[idx]) for idx in sampled_candi_indices[:self.num_hazard]]
            candi_goal_pos_list = [np.array(self.candi_pos_list[idx]) for idx in sampled_candi_indices[self.num_hazard:]]
            good_goal_pos_list = []
            for candi_goal_pos in candi_goal_pos_list:
                is_good = True
                for hazard_pos in hazard_pos_list:
                    hazard_dist = np.linalg.norm(hazard_pos - candi_goal_pos) - self.hazard_size
                    if hazard_dist <= self.limit_distance + 2.0/self.h_coeff:
                        is_good = False
                        break
                if is_good:
                    good_goal_pos_list.append(candi_goal_pos)

            if len(good_goal_pos_list) >= 3:
                min_dist = np.inf
                for goal_idx in range(len(good_goal_pos_list) - 1):
                    dist = np.linalg.norm(good_goal_pos_list[goal_idx] - good_goal_pos_list[goal_idx+1])
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > 2.0:
                    self.candi_goal_pos_list = deepcopy(good_goal_pos_list)
                    self.hazard_pos_list = deepcopy(hazard_pos_list)
                    break
                else:
                    pass
            else:
                pass
        
        for i in range(self.num_hazard):
            candi_pos = self.hazard_pos_list[i]
            self.sim.data.set_joint_qpos('box{}'.format(i+1), [*candi_pos, 0.25, 1.0, 0.0, 0.0, 0.0])
            self.sim.data.set_joint_qvel('box{}'.format(i+1), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        robot_id = self.sim.model.body_name2id('robot')
        self.sim.data.set_joint_qpos('robot', [0.0, 0.0, 0.06344, 1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qvel('robot', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        goal_id = self.sim.model.body_name2id('goal')
        self.sim.data.xfrc_applied[goal_id] = [0.0, 0.0, 0.98, 0.0, 0.0, 0.0]

        self.sim.forward()

    def updateGoalPos(self):
        self.goal_pos = deepcopy(self.candi_goal_pos_list[0])
        self.candi_goal_pos_list = self.candi_goal_pos_list[1:] + self.candi_goal_pos_list[:1]
        self.sim.data.set_joint_qpos('goal', [*self.goal_pos, 0.25, 1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qvel('goal', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.sim.forward()
        self.pre_goal_dist = self.getGoalDist()

    def getGoalDist(self):
        robot_pos = self.sim.data.get_body_xpos('robot').copy()
        return np.sqrt(np.sum(np.square(self.goal_pos - robot_pos[:2])))

    def reset(self):
        self.pre_vel = 0.0
        self.pre_ang_vel = 0.0
        self.action = np.zeros(2)
        self.robot_vel = np.zeros(2)
        self.pre_robot_vel = np.zeros(2)
        self.build()
        self.updateGoalPos()
        state = self.getState()
        self.cur_step = 0
        return self.getFlattenState(state)

    def get_step_wise_cost(self):
        limit_d = self.limit_distance + self.limit_bound
        scan_value = self.getLidar()
        hazard_dist = np.min(scan_value)
        step_wise_cost = limit_d - hazard_dist
        return step_wise_cost

    def step(self, action):
        self.cur_step += 1
        lin_acc = np.clip(action[0], -1.0, 1.0)
        self.action[0] = np.clip(self.action[0] + lin_acc/self.control_freq, 0.0, 1.0)
        weight = 0.8
        self.action[1] = weight*self.action[1] + (1.0 - weight)*np.clip(action[1], -1.0, 1.0)

        target_vel, target_ang_vel = self.action
        for j in range(self.num_time_step):
            self.sim.forward()
            sensor_dict = self.getSensor()
            vel = sensor_dict['velocimeter'][0]
            ang_vel = sensor_dict['gyro'][2]
            acc = (vel - self.pre_vel)/self.time_step
            ang_acc = (ang_vel - self.pre_ang_vel)/self.time_step
            self.pre_vel = deepcopy(vel)
            self.pre_ang_vel = deepcopy(ang_vel)
            cmd = self.p_coeff*(target_vel - vel) + self.d_coeff*(0.0 - acc)
            ang_cmd = self.ang_p_coeff*(target_ang_vel - ang_vel) + self.ang_d_coeff*(0.0 - ang_acc)
            self.sim.data.ctrl[0] = cmd - ang_cmd
            self.sim.data.ctrl[1] = cmd + ang_cmd
            self.sim.step()

        state = self.getState()
        info = {"goal_met":False, 'cost':0.0, 'num_cv':0}

        # reward
        goal_dist = state['goal_dist']
        reward = self.pre_goal_dist - goal_dist
        self.pre_goal_dist = goal_dist
        if goal_dist < self.goal_dist_threshold:
            print("goal met!")
            reward += 1.0
            info['goal_met'] = True
            self.updateGoalPos()

        # cv
        num_cv = 0
        hazard_dist = np.min(state['scan'])
        if hazard_dist < self.limit_distance:
            num_cv += 1
        info['num_cv'] = num_cv
        info['cost'] = self.getCost(hazard_dist)

        # done
        wall_contact = False
        for contact_item in self.sim.data.contact:
            name1 = self.sim.model.geom_id2name(contact_item.geom1)
            name2 = self.sim.model.geom_id2name(contact_item.geom2)
            if name1 is None or name2 is None or name1=='floor' or name2=='floor':
                continue
            if (name1 == 'robot' and ('wall' in name2 or 'box' in name2)) or (name2 == 'robot' and ('wall' in name1 or 'box' in name1)):
                wall_contact = True
                break
        done = False
        if self.cur_step >= self.max_steps or wall_contact:
            done = True
            discount_factor = 0.99
            temp_num_cv = max(self.max_steps - self.cur_step, 0)
            temp_cost = discount_factor*(1 - discount_factor**temp_num_cv)/(1 - discount_factor)
            info['num_cv'] += temp_num_cv
            info['cost'] += temp_cost

        #add raw state
        info['raw_state'] = state

        return self.getFlattenState(state), reward, done, info
