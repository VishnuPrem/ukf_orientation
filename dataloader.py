#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter

import scipy.io as sio
import os
import numpy as np
import math
import matplotlib.pyplot as plt

class DataLoader:
    
    def get_imu_data(self, file_num):
        
        file_name = "imuRaw"+str(file_num)+".mat"
        folder_name = "imu"
        data_dir = os.getcwd() + "\\" + folder_name + "\\" + file_name
        data = sio.loadmat(data_dir)  
            
        ts = np.transpose(data['ts'])
                     
        vals = data['vals']
        vals = vals.astype('float64')
        
        gyro_data = vals[3:]
        gyro_data[[0,1,2]] = gyro_data[[1,2,0]]
        
        gyro_scale_fac = 0.015
        gyro_bias0 = 374
        gyro_bias1 = 375.5
        gyro_bias2 = 370
        
        gyro_data[0] = (gyro_data[0]-gyro_bias0) * gyro_scale_fac
        gyro_data[1] = (gyro_data[1]-gyro_bias1) * gyro_scale_fac
        gyro_data[2] = (gyro_data[2]-gyro_bias2) * gyro_scale_fac
        
        gyro_data = self.integrate(gyro_data, ts)
                
        acc_data = vals[:3]
        acc_scale_fac = 0.09
        acc_bias0 = 500
        acc_bias1 = 500
        
        acc_data[0] = (acc_data[0]-acc_bias0) * acc_scale_fac
        acc_data[1] = (acc_data[1]-acc_bias1) * acc_scale_fac
        acc_data[2] = (acc_data[2]-acc_bias1) * acc_scale_fac
        
        roll = np.arctan2(-acc_data[1], acc_data[2])
        pitch = np.arctan2(acc_data[0], np.sqrt(acc_data[1]*acc_data[1] + acc_data[2]*acc_data[2]))
                
        return gyro_data, [roll, pitch], ts
        
    def get_imu_data_for_filter(self, file_num, Local = False):
        
        if Local == True:
            file_name = "imuRaw"+str(file_num)+".mat"
            folder_name = "imu"
            data_dir = os.getcwd() + "\\" + folder_name + "\\" + file_name
            data = sio.loadmat(data_dir)
        else:
            file_name = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(file_num) + ".mat")
            data = sio.loadmat(file_name) 
        
        ts = np.transpose(data['ts'])
                     
        vals = data['vals']
        vals = vals.astype('float64')
        
        gyro_data = vals[3:]
        gyro_data[[0,1,2]] = gyro_data[[1,2,0]]
        
        gyro_scale_fac = 0.015
        gyro_bias0 = 374
        gyro_bias1 = 375.5
        gyro_bias2 = 370
        
        gyro_data[0] = (gyro_data[0]-gyro_bias0) * gyro_scale_fac
        gyro_data[1] = (gyro_data[1]-gyro_bias1) * gyro_scale_fac
        gyro_data[2] = (gyro_data[2]-gyro_bias2) * gyro_scale_fac
        
        acc_data = vals[:3]
        acc_scale_fac = 0.09
        acc_bias0 = 500
        acc_bias1 = 500
        acc_bias2 = 500
        
        acc_data[0] = (acc_data[0]-acc_bias0) * acc_scale_fac * -1
        acc_data[1] = (acc_data[1]-acc_bias1) * acc_scale_fac * -1
        acc_data[2] = (acc_data[2]-acc_bias2) * acc_scale_fac
        
        return gyro_data, acc_data, ts
    
    # loads vicon data from directory
    def get_vicon_data(self, file_num):
        file_name = "viconRot"+str(file_num)+".mat"
        folder_name = "vicon"
        data_dir = os.getcwd() + "\\" + folder_name + "\\" + file_name
        data = sio.loadmat(data_dir)  
        
        return self.compute_rpy_from_vicon(data['rots']), np.transpose(data['ts'])
    
    
    # computes euler angles from rotation matrices
    def compute_rpy_from_vicon(self, rotations):   
        
        rpyaw = np.zeros((3, rotations.shape[-1]))
        
        for i in range(rotations.shape[-1]):
            
            rot_matrix = rotations[...,i]
            
            yaw = math.atan2(rot_matrix[1,0], rot_matrix[0,0])
            pitch = math.atan2(-rot_matrix[2,0], math.sqrt( (rot_matrix[2,1]**2) + (rot_matrix[2,2]**2)))
            roll = math.atan2(rot_matrix[2,1], rot_matrix[2,2])
            
            rpyaw[0,i] = roll
            rpyaw[1,i] = pitch
            rpyaw[2,i] = yaw          
        return rpyaw
    
    def integrate(self, w, t):
        
        t2 = np.delete(t, 0, 0)
        t1 = np.delete(t, -1, 0)
        
        delta_t = t2-t1
        delta_t = np.append(delta_t, delta_t[-1])
        
        x_dash = w * delta_t
        x0 = np.cumsum(x_dash[0])
        x1 = np.cumsum(x_dash[1])
        x2 = np.cumsum(x_dash[2])
        return np.array([x0,x1,x2])
        
    def plot_graph(self, x, y, third = True):    
        plt.figure(1)
        plt.subplot(311)
        plt.plot(x, y[0])
        plt.subplot(312)
        plt.plot(x, y[1])
        plt.subplot(313)
        if third:
            plt.plot(x, y[2])        
        plt.show()
        
    def plot_both_graph(self, x1, x2, y1, y2, third = True):
        
        x1 = x1.reshape(-1)
        x2 = x2.reshape(-1)
        plt.figure(1)
        plt.subplot(311)
        plt.plot(x1, y1[0], 'r')
        plt.plot(x2, y2[0], 'b')
        plt.subplot(312)
        plt.plot(x1, y1[1], 'r')
        plt.plot(x2, y2[1], 'b')
        plt.subplot(313)
        if third:
            plt.plot(x1, y1[2], 'r')
            plt.plot(x2, y2[2], 'b')
        plt.show()
   
     
def test_vicon_only():    
    d = DataLoader()
    rots, ts = d.get_vicon_data(1)
    d.plot_graph(ts,rots)

def test_acc_only():      
    d = DataLoader()
    gyr, acc, ts = d.get_imu_data(1)
    d.plot_graph(ts, acc, False)

def tune_acc():
    d = DataLoader()
    rots, ts_vicon = d.get_vicon_data(3)
    gyr, acc, ts_imu = d.get_imu_data(3)
    d.plot_both_graph(ts_vicon, ts_imu, rots, acc, False)
    
def tune_gyr():    
    d = DataLoader()
    rots, ts_vicon = d.get_vicon_data(3)
    gyr, acc, ts_imu = d.get_imu_data(3)
    d.plot_both_graph(ts_vicon, ts_imu, rots, gyr)

# tune_gyr()
# tune_acc()