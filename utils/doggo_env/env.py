import tensorflow as tf
import numpy as np
import safety_gym
import pickle
import gym
import os

class Env(gym.Env):
    def __init__(self):
        self.env_name = "Safexp-DoggoGoal1-v0"
        self._env = gym.make(self.env_name)
        self.max_episode_length = 1000
        self.action_repeat = 1
        self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16
        self.action_dim = 2
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = gym.spaces.box.Box(-np.ones(self.action_dim), np.ones(self.action_dim))
        self.goal_threshold = np.inf
        self.hazard_size = 0.2
        self.safety_confidence = 0.0
        self.hazard_size_confidence = self.hazard_size + self.safety_confidence

        # build low-level policy
        self.buildLowPolicy()
        #make session and load model
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
        # set low-level policy session
        self.setSession(sess)
        self.initLowPolicy(sess)


    def seed(self, num_seed):
        self._env.seed(num_seed)

    def buildLowPolicy(self):
        hidden1_units = 512
        hidden2_units = 512
        action_dim = 12

        with tf.variable_scope('LOW_LEVEL'):

            self.states_ph = tf.placeholder(tf.float32, [None, 48], name='states')
            with tf.variable_scope('policy'):
                model = tf.layers.dense(self.states_ph, hidden1_units, activation=tf.nn.relu)
                model = tf.layers.dense(model, hidden2_units, activation=tf.nn.relu)
                self.means = tf.layers.dense(model, action_dim, activation=tf.nn.sigmoid)
                log_std = tf.layers.dense(model, action_dim, activation=None)
                log_std = tf.clip_by_value(log_std, -8, 2)
                self.stds = tf.exp(log_std)

            def unnormalize_action(a):
                action_bound_max = np.ones(12, dtype=np.float32)
                action_bound_min = -np.ones(12, dtype=np.float32)
                temp_a = action_bound_max - action_bound_min
                temp_b = action_bound_min
                temp_a = tf.ones_like(a)*temp_a
                temp_b = tf.ones_like(a)*temp_b
                return temp_a*a + temp_b

            self.dist = tf.distributions.Normal(self.means, self.stds)
            norm_noise_action = self.dist.sample()
            norm_action = self.dist.mode()
            self.sample_noise_action = unnormalize_action(norm_noise_action)
            self.sample_action = unnormalize_action(norm_action)

            policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/policy")
            flatten_vars = tf.concat([tf.reshape(g, [-1]) for g in policy_vars], axis=0)

            self.params_ph = tf.placeholder(tf.float32, flatten_vars.shape, name='params')
            self.assign_op = []
            start = 0
            for var in policy_vars:
                size = np.prod(var.shape)
                param = tf.reshape(self.params_ph[start:start + size], var.shape)
                self.assign_op.append(var.assign(param))
                start += size

    def initLowPolicy(self, sess):
        abs_path = os.path.dirname(__file__)
        with open(f"{abs_path}/policy_net.pkl", "rb") as f:
            policy_vars_value = pickle.load(f)
        sess.run(self.assign_op, feed_dict={self.params_ph:policy_vars_value})
        print("[ENV] initialized low-level policy!")

    def getGoal(self):
        goal_dir = self._env.obs_compass(self._env.goal_pos)
        goal_dist = self._env.dist_goal()
        goal_dist = np.clip(goal_dist, 0.0, 1.0)
        goal = goal_dist*goal_dir
        return goal

    def getInternalState(self, goal):
        goal_dist = np.linalg.norm(goal)
        goal_dir = np.array(goal)/(goal_dist + 1e-8)
        goal_dist = np.clip([goal_dist], 0.0, 1.0)
        vel = self._env.world.get_sensor('velocimeter')
        rot_vel = self._env.world.get_sensor('gyro')
        magneto = self._env.world.get_sensor('magnetometer')
        state = np.concatenate([goal_dir, goal_dist, vel, rot_vel, magneto], axis=0)
        flat_obs = []
        for sensor in self._env.robot.hinge_vel_names:
            flat_obs += list(self._env.world.get_sensor(sensor).flat)
        for sensor in self._env.robot.hinge_pos_names:
            theta = float(self._env.world.get_sensor(sensor))
            flat_obs += list(np.array([np.sin(theta), np.cos(theta)]).flat)
        state = np.concatenate([state, flat_obs], axis=0)
        return state

    def setSession(self, sess):
        self.sess = sess

    def getLowAction(self, goal):
        state = self.getInternalState(goal)
        feed_dict={self.states_ph:[state]}
        [action] = self.sess.run(self.sample_noise_action, feed_dict=feed_dict)
        return action

    def getState(self):
        goal_dir = self._env.obs_compass(self._env.goal_pos)
        goal_dist = np.array([self._env.dist_goal()])
        goal_dist = np.clip(goal_dist, 0.0, self.goal_threshold)
        hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
        acc = self._env.world.get_sensor('accelerometer')[:2]
        vel = self._env.world.get_sensor('velocimeter')[:2]
        rot_vel = self._env.world.get_sensor('gyro')[2:]
        state = np.concatenate([goal_dir/0.7, (goal_dist - 1.5)/0.6, acc/8.0, vel/0.2, rot_vel/2.0, (hazards_lidar - 0.3)/0.3], axis=0)
        return state

    def getCost(self, h_dist):
        h_coeff = 10.0
        cost = 1.0/(1.0 + np.exp((h_dist - self.hazard_size_confidence)*h_coeff))
        return cost

    def getMinDist(self, hazard_pos_list, pos):
        pos = np.array(pos)
        min_dist = np.inf
        for hazard_pos in hazard_pos_list:
            dist = np.linalg.norm(hazard_pos[:2] - pos[:2])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def getHazardsDist(self):
        h_dist = self.getMinDist(self._env.hazards_pos, self._env.world.robot_pos())
        return h_dist

    def get_step_wise_cost(self):
        h_dist = self.getHazardsDist()
        step_wise_cost =  self.hazard_size_confidence - h_dist
        return step_wise_cost
        
    def reset(self):
        self.t = 0
        self._env.reset()
        state = self.getState()
        return state

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        reward = 0
        is_goal_met = False
        num_cv = 0

        for _ in range(self.action_repeat):
            low_action = self.getLowAction(action)
            s_t, r_t, d_t, info = self._env.step(low_action)

            if info['cost'] > 0:
                num_cv += 1

            try:
                if info['goal_met']:
                    is_goal_met = True
            except:
                pass
                
            reward += r_t
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break

        state = self.getState()
        h_dist = self.getHazardsDist()

        info['goal_met'] = is_goal_met
        info['cost'] = self.getCost(h_dist)
        info['num_cv'] = num_cv
        return state, reward, done, info

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()
