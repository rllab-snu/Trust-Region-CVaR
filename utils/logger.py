import numpy as np
import pickle
import glob
import os

class Logger:
    def __init__(self, env_name, save_name):
        self.save_name = save_name
        save_name = '{}/{}_log'.format(env_name, save_name.lower())
        if not os.path.isdir(save_name):
            os.makedirs(save_name)

        exist_list = glob.glob("{}/record_*.pkl".format(save_name))
        record_idx = len(exist_list)
        self.log_name = save_name+"/record_%02d.pkl" % (record_idx)
        self.log = []


    def write(self, data):
        self.log.append(data)

    def save(self):
        with open(self.log_name, 'wb') as f:
            pickle.dump(self.log, f)

    def get_avg(self, length=1):
        length = min(len(self.log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in self.log[-length:]]
        return np.mean(temp_data)

    def get_avg2(self, length=1):
        length = min(len(self.log), length) 
        if length == 0: return None
        temp_data = [item[1]/item[0] for item in self.log[-length:]]
        return np.mean(temp_data)

    def get_std(self, length=1):
        length = min(len(self.log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in self.log[-length:]]
        return np.std(temp_data)

    def get_square(self, length=1):
        length = min(len(self.log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in self.log[-length:]]
        return np.mean(np.square(temp_data))

    def get_cvar(self, sigma_unit, length=1):
        length = min(len(self.log), length) 
        if length == 0: return None
        temp_data = [item[1] for item in self.log[-length:]]
        mean = np.mean(temp_data)
        std = np.std(temp_data)
        cvar = mean + sigma_unit*std
        return cvar
