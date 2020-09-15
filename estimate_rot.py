#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter

import dataloader 
import filter_func as filt
import quaternion
import numpy as np

def estimate_rot(file_num=1, Local = False):   
    
    prior_state = [quaternion.Quaternion(1,0,0,0), np.array([0.1,0.1,0.1])]
    
    # prior process covariance
    Pk = np.identity(6)*1
    Q =  np.identity(6)*0.000001
    R =  np.identity(6)*0.001
    
    data = dataloader.DataLoader()
    
    if Local:
        vicon_data, ts_vicon = data.get_vicon_data(file_num)
        gyr_data, acc_data, ts_imu = data.get_imu_data_for_filter(file_num)
    else:
        gyr_data, acc_data, ts_imu = data.get_imu_data_for_filter(file_num)
        
    # ts_imu = ts_imu[2000:4000]
    # vicon_data = vicon_data[:,2000:4000]
    # ts_vicon = ts_vicon[2000:4000]
    estimated_rpy = np.zeros((3, ts_imu.shape[0]))
    
    for i in range(ts_imu.shape[0]):
        # i=1
        
        sigma_points = filt.get_sigma_points(prior_state, Pk, Q)
        
        transformed_pts = filt.transform_sigma_points(sigma_points, ts_imu[i] - ts_imu[i-1])
        
        avg_pt = filt.average_sigma_points(transformed_pts, prior_state)
        
        Wi = filt.subtract_mean_from_sigma_points(transformed_pts, avg_pt)
        
        Pxx = filt.priori_process_covariance(Wi)
        
        Zi, Z_mean = filt.measurement_model(transformed_pts)
        
        measurement = np.concatenate((acc_data[:,i], gyr_data[:,i]))
        Vk = filt.innovation(measurement, Z_mean)
        
        Pvv, Zi_minus_Zmean = filt.measurement_estimate_covariance(Zi, Z_mean, R)
        
        Pxz = filt.cross_correlation_matrix(Wi, Zi_minus_Zmean)
        
        K = filt.kalman_gain(Pxz, Pvv)
        
        prior_state = filt.postiriori_estimate(K, avg_pt, Vk)
        
        estimated_rpy[:,i] = prior_state[0].euler_angles()    
        
        Pk = filt.covariance_update(Pxx, K, Pvv)
        
        if Local:
            print(i)
    
    if Local:
        return data, ts_vicon, ts_imu, vicon_data, estimated_rpy

    return estimated_rpy[0], estimated_rpy[1], estimated_rpy[2]


data, ts_vicon, ts_imu, vicon_data, estimated_rpy = estimate_rot(1, True)
data.plot_both_graph(ts_vicon, ts_imu, vicon_data, estimated_rpy)

# r,p,y = estimate_rot(1)
