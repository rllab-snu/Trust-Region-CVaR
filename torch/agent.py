from typing import Optional, List

from models import Policy
from models import Value2
from models import Value

from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

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
        self.device = args.device
        self.checkpoint_dir='{}/checkpoint'.format(args.save_dir)

        # for training
        self.discount_factor = args.discount_factor
        self.n_epochs = args.n_epochs
        self.gae_coeff = args.gae_coeff
        self.ent_coeff = args.ent_coeff
        self.lr = args.lr

        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl

        # for cost
        self.cost_d = args.cost_d
        self.cost_alpha = args.cost_alpha
        self.sigma_unit = norm.pdf(norm.ppf(self.cost_alpha))/self.cost_alpha

        # for envs
        self.n_envs = args.n_envs
        self.n_steps = args.n_steps
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device)

        # for model
        self.policy = Policy(args).to(args.device)
        self.value = Value(args).to(args.device)
        self.cost_value = Value(args).to(args.device)
        self.cost_std_value = Value2(args).to(args.device)
        self.cost_var_value = lambda x: torch.square(self.cost_std_value(x))
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.lr)
        self.cost_v_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=self.lr)
        self.cost_std_v_optimizer = torch.optim.Adam(self.cost_std_value.parameters(), lr=self.lr)

        # load
        self.load()


    def normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def getAction(self, state:torch.Tensor, is_train:bool) -> List[torch.Tensor]:
        '''
        input:
            states:     Tensor(state_dim,)
            is_train:   boolean
        output:
            action:         Tensor(action_dim,)
            cliped_action:  Tensor(action_dim,)
        '''
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self.unnormalizeAction(mean + noise*std)
        else:
            action = self.unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)
        return action, clipped_action

    def getGaesTargets(self, rewards:np.ndarray, values:np.ndarray, dones:np.ndarray, fails:np.ndarray, next_values:np.ndarray) -> List[np.ndarray]:
        '''
        input:
            rewards:        np.array(n_steps,)
            values:         np.array(n_steps,)
            dones:          np.array(n_steps,)
            fails:          np.array(n_steps,)
            next_values:    np.array(n_steps,)
        output:
            gaes:       np.array(n_steps,)
            targets:    np.array(n_steps,)
        '''
        deltas = rewards + (1.0 - fails)*self.discount_factor*next_values - values
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t])*self.discount_factor*self.gae_coeff*gaes[t + 1]
        targets = values + gaes
        return gaes, targets

    def getVarGaesTargets(self, costs:np.ndarray, values:np.ndarray, var_values:np.ndarray, dones:np.ndarray, fails:np.ndarray, next_values:np.ndarray, next_var_values:np.ndarray) -> List[np.ndarray]:
        '''
        input:
            costs:              np.array(n_steps,)
            values:             np.array(n_steps,)
            var_values:         np.array(n_steps,)
            dones:              np.array(n_steps,)
            fails:              np.array(n_steps,)
            next_values:        np.array(n_steps,)
            next_var_values:    np.array(n_steps,)
        output:
            gaes:       np.array(n_steps,)
            targets:    np.array(n_steps,)
        '''
        deltas = np.square(costs + (1.0 - fails)*self.discount_factor*next_values) - np.square(values) + \
                                    (1.0 - fails)*(self.discount_factor**2)*next_var_values - var_values
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t])*((self.discount_factor*self.gae_coeff)**2)*gaes[t + 1]
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

            # convert to tensor
            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)

            # get GAEs and Tagets
            # for reward
            values_tensor = self.value(states_tensor)
            next_values_tensor = self.value(next_states_tensor)
            values = values_tensor.detach().cpu().numpy()
            next_values = next_values_tensor.detach().cpu().numpy()
            gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)
            # for cost
            cost_values_tensor = self.cost_value(states_tensor)
            next_cost_values_tensor = self.cost_value(next_states_tensor)
            cost_values = cost_values_tensor.detach().cpu().numpy()
            next_cost_values = next_cost_values_tensor.detach().cpu().numpy()
            cost_gaes, cost_targets = self.getGaesTargets(costs, cost_values, dones, fails, next_cost_values)
            # for cost var
            cost_var_values_tensor = self.cost_var_value(states_tensor)
            next_cost_var_values_tensor = self.cost_var_value(next_states_tensor)
            cost_var_values = cost_var_values_tensor.detach().cpu().numpy()
            next_cost_var_values = next_cost_var_values_tensor.detach().cpu().numpy()
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
        cost_mean = np.mean(cost_means_list)
        cost_var_mean = np.mean(cost_var_targets)

        # convert to tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float32)
        norm_actions_tensor = self.normalizeAction(actions_tensor)
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float32)
        cost_gaes_tensor = torch.tensor(cost_gaes, device=self.device, dtype=torch.float32)
        cost_targets_tensor = torch.tensor(cost_targets, device=self.device, dtype=torch.float32)
        cost_square_gaes_tensor = torch.tensor(cost_square_gaes, device=self.device, dtype=torch.float32)
        cost_std_targets_tensor = torch.tensor(np.sqrt(cost_var_targets), device=self.device, dtype=torch.float32)

        # ==================== for policy update ==================== #
        # backup old policy
        means, log_stds, stds = self.policy(states_tensor)
        old_means = means.clone().detach()
        old_stds = stds.clone().detach()

        # get objective & KL & cost surrogate
        objective, entropy = self.getObjective(states_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
        cost_surrogate = self.getCostSurrogate(states_tensor, norm_actions_tensor, old_means, \
                            old_stds, cost_gaes_tensor, cost_square_gaes_tensor, cost_mean, cost_var_mean)
        kl = self.getKL(states_tensor, old_means, old_stds)

        # get gradient
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        grad_b = flatGrad(-cost_surrogate, self.policy.parameters(), retain_graph=True)
        x_value = self.conjugateGradient(kl, grad_g)
        approx_g = self.Hx(kl, x_value)
        cost_d = self.cost_d/(1.0 - self.discount_factor)
        c_value = cost_surrogate - cost_d

        # ======== solve Lagrangian problem ======== #
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
        if torch.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = torch.dot(approx_g, x_value)
            optim_case = 4
        else:
            H_inv_b = self.conjugateGradient(kl, grad_b)
            approx_b = self.Hx(kl, H_inv_b)
            scalar_q = torch.dot(approx_g, x_value)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)
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
            lam = torch.sqrt(scalar_q/(2*self.max_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value/B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q/(2*self.max_kl)), LB)
            f_a = lambda lam : -0.5 * (A_value / (lam + EPS) + B_value * lam) - scalar_r*c_value/(scalar_s + EPS)
            f_b = lambda lam : -0.5 * (scalar_q / (lam + EPS) + 2*self.max_kl*lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c_value - scalar_r) / (scalar_s + EPS)
        else:
            lam = 0
            nu = torch.sqrt(2*self.max_kl / (scalar_s+EPS))
        # ========================================== #

        # line search
        delta_theta = (1./(lam + EPS))*(x_value + nu*H_inv_b) if optim_case > 0 else nu*H_inv_b
        beta = 1.0
        init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        init_cost_surrogate = cost_surrogate.clone().detach()
        while True:
            theta = beta*delta_theta + init_theta
            self.applyParams(theta)
            objective, entropy = self.getObjective(states_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
            cost_surrogate = self.getCostSurrogate(states_tensor, norm_actions_tensor, old_means, \
                                old_stds, cost_gaes_tensor, cost_square_gaes_tensor, cost_mean, cost_var_mean)
            kl = self.getKL(states_tensor, old_means, old_stds)
            if kl <= self.max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                break
            beta *= self.line_decay
        # =========================================================== #

        # ===================== for value update ===================== #
        for _ in range(self.n_epochs):
            value_loss = torch.mean(0.5*torch.square(self.value(states_tensor) - targets_tensor))
            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

            cost_value_loss = torch.mean(0.5*torch.square(self.cost_value(states_tensor) - cost_targets_tensor))
            self.cost_v_optimizer.zero_grad()
            cost_value_loss.backward()
            self.cost_v_optimizer.step()

            cost_var_value_loss = torch.mean(0.5*torch.square(self.cost_std_value(states_tensor) - cost_std_targets_tensor))
            self.cost_std_v_optimizer.zero_grad()
            cost_var_value_loss.backward()
            self.cost_std_v_optimizer.step()
        # ============================================================ #

        scalar = lambda x:x.item()
        value_loss = scalar(value_loss)
        cost_value_loss = scalar(cost_value_loss)
        cost_var_value_loss = scalar(cost_var_value_loss)
        objective = scalar(objective)
        cost_surrogate = scalar(cost_surrogate)
        kl = scalar(kl)
        entropy = scalar(entropy)
        return value_loss, cost_value_loss, cost_var_value_loss, objective, cost_surrogate, kl, entropy, optim_case

    def getObjective(self, states, norm_actions, gaes, old_means, old_stds):
        means, _, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds + EPS)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        entropy = torch.mean(torch.sum(dist.entropy(), dim=1))

        old_dist = torch.distributions.Normal(old_means, old_stds + EPS)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)

        objective = torch.mean(torch.exp(log_probs - old_log_probs)*gaes)
        objective += self.ent_coeff*(entropy/self.action_dim)
        return objective, entropy

    def getCostSurrogate(self, states, norm_actions, old_means, old_stds, cost_gaes, cost_square_gaes, cost_mean, cost_var_mean):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)

        approx_cost_mean = cost_mean + (1.0/(1.0 - self.discount_factor))*(torch.mean(torch.exp(log_probs - old_log_probs)*cost_gaes))
        apprx_cost_var = cost_var_mean + (1.0/(1.0 - self.discount_factor**2))*(torch.mean(torch.exp(log_probs - old_log_probs)*cost_square_gaes))
        cost_surrogate = approx_cost_mean + self.sigma_unit*torch.sqrt(torch.clamp(apprx_cost_var, EPS, np.inf))
        return cost_surrogate

    def getKL(self, states, old_means, old_stds):
        means, log_stds, stds = self.policy(states)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl

    def applyParams(self, params):
        n = 0
        for p in self.policy.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def Hx(self, kl:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        '''
        get (Hessian of KL * x).
        input:
            kl: tensor(,)
            x: tensor(dim,)
        output:
            Hx: tensor(dim,)
        '''
        flat_grad_kl = flatGrad(kl, self.policy.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.policy.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def conjugateGradient(self, kl:torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        '''
        get (H^{-1} * g).
        input:
            kl: tensor(,)
            g: tensor(dim,)
        output:
            H^{-1}g: tensor(dim,)
        '''
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(self.num_conjugate):
            Ap = self.Hx(kl, p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/rs_old)*p
            rs_old = rs_new
        return x

    def save(self):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'cost_value': self.cost_value.state_dict(),
            'cost_std_value': self.cost_std_value.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'cost_v_optimizer': self.cost_v_optimizer.state_dict(),
            'cost_std_v_optimizer': self.cost_std_v_optimizer.state_dict(),
            }, f"{self.checkpoint_dir}/model.pt")
        print('[save] success.')

    def load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.policy.load_state_dict(checkpoint['policy'])
            self.value.load_state_dict(checkpoint['value'])
            self.cost_value.load_state_dict(checkpoint['cost_value'])
            self.cost_std_value.load_state_dict(checkpoint['cost_std_value'])
            self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
            self.cost_v_optimizer.load_state_dict(checkpoint['cost_v_optimizer'])
            self.cost_std_v_optimizer.load_state_dict(checkpoint['cost_std_v_optimizer'])
            print('[load] success.')
        else:
            self.policy.initialize()
            self.value.initialize()
            self.cost_value.initialize()
            self.cost_std_value.initialize()
            print('[load] fail.')
