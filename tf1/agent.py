import tensorflow_probability as tfp
from scipy.stats import norm
from copy import deepcopy
import tensorflow as tf
import numpy as np
import pickle
import random
import time
import os

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

class Agent:
    def __init__(self, args):
        '''
        We implement square cost value networks as followings:
        1. define the following network (cost std, cost variance).
            Variance_c^{\pi}(s) := Square_c^{\pi}(s) - Value_c^{\pi}(s)**2
                                 = Std_c^{\pi}(s)**2
        2. express the square cost value using the std and varaince networks.
            Square_c^{\pi}(s) := Value_c^{\pi}(s)**2 + Std_c^{\pi}(s)**2
        '''
        # base
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'
        # for env
        self.discount_factor = args.discount_factor
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = args.action_bound_min
        self.action_bound_max = args.action_bound_max
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps
        # for networks
        self.hidden_dim = args.hidden_dim
        self.log_std_init = args.log_std_init
        self.activ_func = eval(f"tf.nn.{args.activation}")
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.gae_coeff = args.gae_coeff
        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl
        # for cost
        self.cost_d = args.cost_d
        self.cost_alpha = args.cost_alpha
        self.sigma_unit = norm.pdf(norm.ppf(self.cost_alpha))/self.cost_alpha

        with tf.variable_scope(self.name):
            #placeholder
            self.states_ph = tf.placeholder(tf.float32, [None, self.state_dim], name='states')
            self.actions_ph = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')
            self.targets_ph = tf.placeholder(tf.float32, [None,], name='targets')
            self.cost_targets_ph = tf.placeholder(tf.float32, [None,], name='cost_targets')
            self.cost_var_targets_ph = tf.placeholder(tf.float32, [None,], name='cost_var_targets')
            self.gaes_ph = tf.placeholder(tf.float32, [None,], name='gaes')
            self.cost_gaes_ph = tf.placeholder(tf.float32, [None,], name='cost_gaes')
            self.cost_square_gaes_ph = tf.placeholder(tf.float32, [None,], name='cost_square_gaes')
            self.old_means_ph = tf.placeholder(tf.float32, [None, self.action_dim], name='old_means')
            self.old_stds_ph = tf.placeholder(tf.float32, [None, self.action_dim], name='old_stds')
            self.cost_mean_ph = tf.placeholder(tf.float32, (), name='cost_mean')
            self.cost_var_mean_ph = tf.placeholder(tf.float32, (), name='cost_var_mean')

            # policy
            self.means, self.stds = self.policyModel(self.states_ph, 'policy')
            self.dist = tfp.distributions.Normal(self.means, self.stds)
            self.old_dist = tfp.distributions.Normal(self.old_means_ph, self.old_stds_ph)
            self.entropy = tf.reduce_mean(tf.reduce_sum(self.dist.entropy(), axis=1))
            self.kl = tf.reduce_mean(tf.reduce_sum(self.old_dist.kl_divergence(self.dist), axis=1))

            # value
            self.value = self.valueModel(self.states_ph, 'value')
            self.cost_value = self.valueModel(self.states_ph, 'cost_value')
            self.cost_std_value = tf.nn.softplus(self.valueModel(self.states_ph, 'cost_std_value'))
            self.cost_var_value = tf.square(self.cost_std_value)

            #action
            norm_actions_ph = self.normalizeAction(self.actions_ph)
            noise_action = self.dist.sample()
            mean_action = self.dist.mean()
            self.unnorm_noise_action = self.unnormalizeAction(noise_action)
            self.unnorm_mean_action = self.unnormalizeAction(mean_action)

            #value loss
            v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
            self.v_loss = 0.5*tf.square(self.targets_ph - self.value)
            self.v_loss = tf.reduce_mean(self.v_loss)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.v_gradients = tf.gradients(self.v_loss, v_vars)
            self.v_train_op = v_optimizer.apply_gradients(zip(self.v_gradients, v_vars))

            #cost_value loss
            cost_v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/cost_value')
            self.cost_v_loss = 0.5*tf.square(self.cost_targets_ph - self.cost_value)
            self.cost_v_loss = tf.reduce_mean(self.cost_v_loss)
            cost_v_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.cost_v_gradients = tf.gradients(self.cost_v_loss, cost_v_vars)
            self.cost_v_train_op = cost_v_optimizer.apply_gradients(zip(self.cost_v_gradients, cost_v_vars))

            #cost std value loss
            cost_std_v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/cost_std_value')
            self.cost_std_v_loss = 0.5*tf.square(tf.sqrt(self.cost_var_targets_ph) - self.cost_std_value)
            self.cost_std_v_loss = tf.reduce_mean(self.cost_std_v_loss)
            cost_std_v_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.cost_std_v_gradients = tf.gradients(self.cost_std_v_loss, cost_std_v_vars)
            self.cost_std_v_train_op = cost_std_v_optimizer.apply_gradients(zip(self.cost_std_v_gradients, cost_std_v_vars))

            #policy optimizer
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            log_probs = tf.reduce_sum(self.dist.log_prob(norm_actions_ph), axis=1)
            old_log_probs = tf.reduce_sum(self.old_dist.log_prob(norm_actions_ph), axis=1)
            prob_ratios = tf.exp(log_probs - old_log_probs)
            self.objective = tf.reduce_mean(prob_ratios*self.gaes_ph)
            self.grad_g = tf.gradients(self.objective, p_vars)
            self.grad_g = tf.concat([tf.reshape(g, [-1]) for g in self.grad_g], axis=0)

            #cost CVaR
            self.approx_cost_mean = self.cost_mean_ph + (1.0/(1.0 - self.discount_factor))*tf.reduce_mean(prob_ratios*self.cost_gaes_ph)
            self.approx_cost_var = self.cost_var_mean_ph + (1.0/(1.0 - self.discount_factor**2))*tf.reduce_mean(prob_ratios*self.cost_square_gaes_ph)
            self.cost_surrogate = self.approx_cost_mean + self.sigma_unit*tf.sqrt(tf.clip_by_value(self.approx_cost_var, EPS, np.inf))
            self.grad_b = tf.gradients(-self.cost_surrogate, p_vars)
            self.grad_b = tf.concat([tf.reshape(b, [-1]) for b in self.grad_b], axis=0)

            # hessian
            kl_grad = tf.gradients(self.kl, p_vars)
            kl_grad = tf.concat([tf.reshape(g, [-1]) for g in kl_grad], axis=0)
            self.theta_ph = tf.placeholder(tf.float32, shape=kl_grad.shape, name='theta')
            self.hessian_product = tf.gradients(tf.reduce_sum(kl_grad*self.theta_ph), p_vars)
            self.hessian_product = tf.concat([tf.reshape(g, [-1]) for g in self.hessian_product], axis=0)
            self.hessian_product += self.damping_coeff * self.theta_ph

            # for line search
            self.flatten_p_vars = tf.concat([tf.reshape(g, [-1]) for g in p_vars], axis=0)
            self.params = tf.placeholder(tf.float32, self.flatten_p_vars.shape, name='params')
            self.assign_op = []
            start = 0
            for p_var in p_vars:
                size = np.prod(p_var.shape)
                param = tf.reshape(self.params[start:start + size], p_var.shape)
                self.assign_op.append(p_var.assign(param))
                start += size

            #make session and load model
            config = tf.ConfigProto()
            ncpu = 2
            config = tf.ConfigProto(inter_op_parallelism_threads=ncpu,
                   intra_op_parallelism_threads=ncpu)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.load()


    def Hx(self, theta, feed_inputs):
        feed_inputs[self.theta_ph] = theta
        return self.sess.run(self.hessian_product, feed_dict=feed_inputs)

    def policyModel(self, states, name='policy', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        param_initializer2 = lambda : tf.random_normal_initializer(mean=self.log_std_init, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            model = tf.layers.dense(states, self.hidden_dim, activation=self.activ_func, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.dense(model, self.hidden_dim, activation=self.activ_func, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            mean = tf.layers.dense(model, self.action_dim, activation=tf.nn.sigmoid, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            log_std = tf.layers.dense(model, self.action_dim, activation=None, bias_initializer=param_initializer2(), kernel_initializer=param_initializer())
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
        return mean, std

    def valueModel(self, states, name='value', reuse=False):
        param_initializer = lambda : tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope(name, reuse=reuse):
            model = states
            model = tf.layers.dense(model, self.hidden_dim, activation=self.activ_func, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.dense(model, self.hidden_dim, activation=self.activ_func, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.layers.dense(model, 1, activation=None, bias_initializer=param_initializer(), kernel_initializer=param_initializer())
            model = tf.reshape(model, [-1])
            return model

    def normalizeAction(self, a):
        temp_a = 1.0/(self.action_bound_max - self.action_bound_min)
        temp_b = self.action_bound_min/(self.action_bound_min - self.action_bound_max)
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b

    def unnormalizeAction(self, a):
        temp_a = self.action_bound_max - self.action_bound_min
        temp_b = self.action_bound_min
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b

    def getAction(self, state, is_train):
        feed_dict={self.states_ph:[state]}
        [noise_action], [action] = self.sess.run([self.unnorm_noise_action, self.unnorm_mean_action], feed_dict=feed_dict)
        if is_train:
            action = noise_action
        clipped_action = np.clip(action, self.action_bound_min, self.action_bound_max)
        return action, clipped_action

    def getActions(self, states, is_train):
        feed_dict={self.states_ph:states}
        noise_actions, actions = self.sess.run([self.unnorm_noise_action, self.unnorm_mean_action], feed_dict=feed_dict)
        if is_train:
            actions = noise_actions
        clipped_actions = np.clip(actions, self.action_bound_min, self.action_bound_max)
        return actions, clipped_actions

    def getGaesTargets(self, rewards, values, dones, fails, next_values):
        deltas = np.array(rewards) + (1.0 - np.array(fails))*self.discount_factor*np.array(next_values) - np.array(values)
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t])*self.discount_factor*self.gae_coeff*gaes[t + 1]
        targets = np.array(values) + np.array(gaes)
        return gaes, targets

    def getVarGaesTargets(self, rewards, values, var_values, dones, fails, next_values, next_var_values):
        deltas = np.square(np.array(rewards) + (1.0 - np.array(fails))*self.discount_factor*np.array(next_values)) - np.square(values) + (1.0 - np.array(fails))*(self.discount_factor**2)*np.array(next_var_values) - np.array(var_values)
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t])*(self.discount_factor**2)*self.gae_coeff*gaes[t + 1]
        targets = np.clip(np.array(var_values) + np.array(gaes), 0.0, np.inf)
        return gaes, targets

    def train(self, trajs):
        states_list = []
        actions_list = []
        gaes_list = []
        cost_gaes_list = []
        cost_square_gaes_list = []
        targets_list = []
        cost_targets_list = []
        cost_var_targets_list = []
        cost_means_list = []

        # to calculate GAEs and targets
        for env_idx in range(self.n_envs):
            states = np.array([traj[0] for traj in trajs[env_idx]])
            actions = np.array([traj[1] for traj in trajs[env_idx]])
            rewards = np.array([traj[2] for traj in trajs[env_idx]])
            costs = np.array([traj[3] for traj in trajs[env_idx]])
            dones = np.array([traj[4] for traj in trajs[env_idx]])
            fails = np.array([traj[5] for traj in trajs[env_idx]])
            next_states = np.array([traj[6] for traj in trajs[env_idx]])

            values, cost_values, cost_var_values = self.sess.run([self.value, self.cost_value, self.cost_var_value], feed_dict={self.states_ph:states})
            next_values, next_cost_values, next_cost_var_values = self.sess.run([self.value, self.cost_value, self.cost_var_value], feed_dict={self.states_ph:next_states})

            gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)
            cost_gaes, cost_targets = self.getGaesTargets(costs, cost_values, dones, fails, next_cost_values)
            cost_square_gaes, cost_var_targets = self.getVarGaesTargets(costs, cost_values, cost_var_values, dones, fails, next_cost_values, next_cost_var_values)

            states_list.append(states)
            actions_list.append(actions)
            gaes_list.append(gaes)
            cost_gaes_list.append(cost_gaes)
            cost_square_gaes_list.append(cost_square_gaes)
            targets_list.append(targets)
            cost_targets_list.append(cost_targets)
            cost_var_targets_list.append(cost_var_targets)
            cost_means_list.append(np.mean(costs)/(1.0 - self.discount_factor))

        states = np.concatenate(states_list)
        actions = np.concatenate(actions_list)
        gaes = np.concatenate(gaes_list)
        cost_gaes = np.concatenate(cost_gaes_list)
        cost_square_gaes = np.concatenate(cost_square_gaes_list)
        targets = np.concatenate(targets_list)
        cost_targets = np.concatenate(cost_targets_list)
        cost_var_targets = np.concatenate(cost_var_targets_list)

        gaes = (gaes - np.mean(gaes))/(np.std(gaes) + EPS)
        cost_gaes -= np.mean(cost_gaes)
        cost_square_gaes -= np.mean(cost_square_gaes)

        entropy = self.sess.run(self.entropy, feed_dict={self.states_ph:states})
        old_means, old_stds = self.sess.run([self.means, self.stds], feed_dict={self.states_ph:states, self.actions_ph:actions})
        cost_mean = np.mean(cost_means_list)
        cost_var_mean = np.mean(cost_var_targets)

        # POLICY update
        feed_dict = {
            self.states_ph:states,
            self.actions_ph:actions,
            self.gaes_ph:gaes,
            self.cost_gaes_ph:cost_gaes,
            self.cost_square_gaes_ph:cost_square_gaes,
            self.old_means_ph:old_means,
            self.old_stds_ph:old_stds,
            self.cost_mean_ph:cost_mean,
            self.cost_var_mean_ph:cost_var_mean,
        }
        grad_g, grad_b, kl, objective, cost_surrogate = self.sess.run([self.grad_g, self.grad_b, self.kl, self.objective, self.cost_surrogate], feed_dict=feed_dict)
        feed_inputs = {self.states_ph:states, self.old_means_ph:old_means, self.old_stds_ph:old_stds}
        x_value = self.conjugateGradient(grad_g, feed_inputs)
        approx_g = self.Hx(x_value, feed_inputs)

        # ========== solve lagrangian dual problem ========== #
        '''
        The follwing code is from "https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L270".
        Below is the license of the repository.

        MIT License

        Copyright (c) 2019 OpenAI

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        '''
        cost_d = self.cost_d/(1.0 - self.discount_factor)
        c_value = cost_surrogate - cost_d
        if np.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = np.inner(approx_g, x_value)
            optim_case = 4
        else:
            H_inv_b = self.conjugateGradient(grad_b, feed_inputs)
            approx_b = self.Hx(H_inv_b, feed_inputs)
            scalar_q = np.inner(approx_g, x_value)
            scalar_r = np.inner(approx_g, H_inv_b)
            scalar_s = np.inner(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r**2 / scalar_s
            B_value = 2*self.max_kl - c_value**2 / scalar_s
            if c_value < 0 and B_value < 0:
                optim_case = 3
            elif c_value < 0 and B_value >= 0:
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                optim_case = 1
            else:
                optim_case = 0
        if optim_case in [3,4]:
            lam = np.sqrt(scalar_q / (2*self.max_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A_value/B_value), LA)
            lam_b = proj(np.sqrt(scalar_q/(2*self.max_kl)), LB)
            f_a = lambda lam : -0.5 * (A_value / (lam + EPS) + B_value * lam) - scalar_r*c_value/(scalar_s + EPS)
            f_b = lambda lam : -0.5 * (scalar_q / (lam + EPS) + 2*self.max_kl*lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c_value - scalar_r) / (scalar_s + EPS)
        else:
            lam = 0
            nu = np.sqrt(2*self.max_kl / (scalar_s+EPS))
        # =================================================== #

        # line search
        delta_theta = (1./(lam + EPS))*(x_value + nu*H_inv_b) if optim_case > 0 else nu*H_inv_b
        init_theta = self.sess.run(self.flatten_p_vars)
        beta = 1.0
        init_objective, init_cost_surrogate = self.sess.run([self.objective, self.cost_surrogate], feed_dict=feed_dict)
        while True:
            theta = beta*delta_theta + init_theta
            self.sess.run(self.assign_op, feed_dict={self.params:theta})
            kl, objective, cost_surrogate = self.sess.run([self.kl, self.objective, self.cost_surrogate], feed_dict=feed_dict)
            if kl <= self.max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                break
            beta *= self.line_decay

        # VALUE update
        feed_dict = {
            self.states_ph:states, 
            self.targets_ph:targets, 
            self.cost_targets_ph:cost_targets,
            self.cost_var_targets_ph:cost_var_targets, 
        }
        for _ in range(self.n_epochs):
            self.sess.run([self.v_train_op, self.cost_v_train_op, self.cost_std_v_train_op], feed_dict=feed_dict)
        v_loss, cost_v_loss, cost_var_v_loss = self.sess.run([self.v_loss, self.cost_v_loss, self.cost_std_v_loss], feed_dict=feed_dict)
        return v_loss, cost_v_loss, cost_var_v_loss, objective, cost_surrogate, kl, entropy, optim_case

    def conjugateGradient(self, g, feed_inputs):
        x_value = np.zeros_like(g)
        residue = deepcopy(g)
        p_vector = deepcopy(g)
        rs_old = np.inner(residue, residue)
        for i in range(self.num_conjugate):
            Ap = self.Hx(p_vector, feed_inputs)
            pAp = np.inner(p_vector, Ap)
            alpha = rs_old / (pAp + EPS)
            x_value += alpha * p_vector
            residue -= alpha * Ap
            rs_new = np.inner(residue, residue)
            p_vector = residue + (rs_new / rs_old) * p_vector
            rs_old = rs_new
        return x_value

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print(f'[{self.name}] save success.')

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(f'[{self.name}] load success.')
        else:
            self.sess.run(tf.global_variables_initializer())
            print(f'[{self.name}] load fail.')
