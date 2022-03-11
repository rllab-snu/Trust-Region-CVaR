import numpy as np
import pickle
import sys
import os

class RunningMeanStd(object):
    def __init__(self, save_dir, state_dim):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param state_dim: (int) the state_dim of the data stream's output
        """
        self.file_name = f"{save_dir}/normalize.pkl"
        if os.path.isfile(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.mean, self.var, self.count = pickle.load(f)
        else:
            self.mean = np.zeros(state_dim, np.float32)
            self.var = np.ones(state_dim, np.float32)
            self.count = 0.0 #1e-4

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, observations):
        return (observations - self.mean)/np.sqrt(self.var + 1e-8)

    def save(self):
        with open(self.file_name, 'wb') as f:
            pickle.dump([self.mean, self.var, self.count], f)
