
import numpy as np
import quaternion

def process_model(state, ts):
    # state: list [quat, [3]]
    # ts: scalar
    # noise:  list [quat, [3]]
    
    # computing diff quaternion caused by prior angular velocity
    state_w = state[1]
    
    magn_state_w = np.linalg.norm(state_w)
    angle = magn_state_w * ts
    if angle == 0:
        diff_q = quaternion.Quaternion(1,0,0,0)
        
    else:
        axis = state_w / magn_state_w
        diff_q = quaternion.Quaternion.angleaxis_to_quat(angle, axis)
       
    # compute state orientation quaternion
    state_q =  state[0]
    
    # compute new state orientation    
        
    state_q = state_q * diff_q
    
    return [state_q, state_w]
    
    
def get_sigma_points(state, covariance, Q):
    # state: [ quat, [3]]
    # covariance: n x n
    # Q: n x n
    
    n = covariance.shape[0]    
    covariance = covariance + Q
    S = np.linalg.cholesky(covariance)
    
    S1 = S * np.sqrt(2 * n)
    S2 = S * np.sqrt(2 * n) * (-1)
    
    sigma_points = np.concatenate((S1,S2), axis = 1)
    
    sigma_points_list = []
    
    for i in range(sigma_points.shape[-1]):
        sp = sigma_points[:,i]
        sp_quat = quaternion.Quaternion.magnaxis_to_quat(sp[:3])
        sp_quat = state[0] * sp_quat
        sp_ang = state[1] + sp[3:]
        sp_ = [sp_quat, sp_ang]
        sigma_points_list.append(sp_)
    
    # sigma_points_list: [[quat,[3]], [quat,[3]], [quat,[3]] ...] 
    return sigma_points_list
    

def transform_sigma_points(sigma_points_list, ts):
    # sigma_points_list: [[quat,[3]], [quat,[3]], [quat,[3]] ...]
    
    transformed_sigma_points = []
    for sp in sigma_points_list:
        transformed_sp = process_model(sp, ts)
        transformed_sigma_points.append(transformed_sp)
        
    return transformed_sigma_points
   
    
def average_sigma_points(sigma_points_list, state):
    # sigma_points_list: [[quat,[3]], [quat,[3]], [quat,[3]] ...]
    
    quats = []
    ang_vel_matrix = np.zeros((3,len(sigma_points_list)))
    
    for i,sp in enumerate(sigma_points_list):
        quats.append(sp[0])
        ang_vel_matrix[:,i] = sp[1]
    
    mean_quat = quaternion.Quaternion.average_quats(quats, state[0])
    mean_ang_vel = np.average(ang_vel_matrix, axis = 1)
    
    return [mean_quat, mean_ang_vel]


def subtract_mean_from_sigma_points(sigma_points_list, mean_sigma_point):
    # sigma_points_list: [[quat,[3]], [quat,[3]], [quat,[3]] ...]
    # mean_sigma_point: [quat,[3]]
    
    mean_q = mean_sigma_point[0]
    mean_angvel = mean_sigma_point[1]
    mean_q.inverse()
    
    subtracted_sigma_points = np.zeros((6,12))
    
    for i,sp in enumerate(sigma_points_list):
        
        sigma_point_q = sp[0]
        subtracted_q = sigma_point_q * mean_q
        magnmaxis_subtracted_q = subtracted_q.quat_to_magnaxis()
        
        sigma_point_angvel = sp[1]
        subtracted_ang_vel =  sigma_point_angvel -  mean_angvel
        
        subtracted_sigma_point = np.concatenate((magnmaxis_subtracted_q, subtracted_ang_vel))
        subtracted_sigma_points[:,i] = subtracted_sigma_point
        
    # subtracted_sigma_points: [6 x 12] aka Wi
    return subtracted_sigma_points

# Px_x 
def priori_process_covariance(subtracted_sigma_points):
    # subtracted_sigma_points: [6 x 12]
    
    covariance = np.zeros((6,6,12))
    
    for i in range(12):
        sp = subtracted_sigma_points[:,i]
        sp = np.reshape(sp, (6,1))
        sp_T = np.reshape(sp, (1,6))
        covariance[:,:,i] = np.matmul(sp, sp_T)
   
    covariance = np.sum(covariance, axis = 2)
    covariance /= 12
    
    # covariance: [6 x 6]
    return covariance

def measurement_model(sigma_points_list):
    # sigma_points_list: [[quat,[3]], [quat,[3]], [quat,[3]] ...]
    
    measurement_prediction_list = np.zeros((6,12))
    gravity_q = quaternion.Quaternion(0., 0., 0., 0.) 
    gravity_q.q[3] = 9.8
    
    for i, sp in enumerate(sigma_points_list):
        
        sigma_point_q = sp[0]
        
        sigma_point_q.inverse()
        measurement_q = sigma_point_q * gravity_q
        sigma_point_q.inverse()
        measurement_q = measurement_q * sigma_point_q
        
        # measurement_q_magnaxis = measurement_q.quat_to_magnaxis()
        measurement_q_magnaxis = measurement_q.q[1:]
        measurement_prediction = np.concatenate((measurement_q_magnaxis, sp[1]))
        measurement_prediction_list[:,i] = measurement_prediction
        
    avg_measurement = np.average(measurement_prediction_list, axis = 1)
    
    # measurement_prediction_list: [6 x 12] aka Zi
    # avg_measurement_prediction: [6]
    return measurement_prediction_list, avg_measurement


def innovation(measurement, avg_measurement_prediction):
    return measurement - avg_measurement_prediction

# P_zz
def measurement_estimate_covariance(measurement_prediction_list, avg_measurement_prediction, R):
    # measurement_prediction_list: [6 x 12]
    # avg_measurement_prediction: [6]
    avg_measurement_prediction = np.reshape(avg_measurement_prediction, (6,1))
    subtracted_measurement = measurement_prediction_list - avg_measurement_prediction
        
    covariance = np.zeros((6,6,12))
     
    for i in range(12):
        pred= subtracted_measurement[:,i]
        pred = np.reshape(pred, (6,1))
        pred_T = np.reshape(pred, (1,6))
        covariance[:,:,i] = np.matmul(pred, pred_T)
   
    covariance = np.sum(covariance, axis = 2)
    covariance /= 12
    
    covariance += R
        
    # covariance: [6x6]
    # subtracted_measurement: [6x12]
    return covariance, subtracted_measurement

# P_xz
def cross_correlation_matrix(W, Z_minus_mean):
    # W: [6 x 12]
    # Z_minus_mean: [6 x 12]
    
    covariance = np.zeros((6,6,12))
     
    for i in range(12):
        Wi = W[:,i]
        Wi = np.reshape(Wi, (6,1))
        Z_minus_meani_T = np.reshape(Z_minus_mean[:,i], (1,6))
        covariance[:,:,i] = np.matmul(Wi, Z_minus_meani_T)
   
    covariance = np.sum(covariance, axis = 2)
    covariance /= 12
    
    # covariance: [6x6]
    return covariance

def kalman_gain(P_xz, P_vv):
    
    invP_vv = np.linalg.inv(P_vv)
    K = np.matmul(P_xz, invP_vv)   
    return K

def postiriori_estimate(K, mean_state, innovation):
    # K: [6x6]
    # mean_state: [quat,[3]]
    # innovation: [6] 
    
    innovation = np.reshape(innovation, (6,1))
    update = np.matmul(K, innovation)
    update = np.reshape(update,(6))
    update_orientation = update[:3]
    update_w = update[3:]
    
    update_q = quaternion.Quaternion.magnaxis_to_quat(update_orientation)
    
    estimate_q = mean_state[0] * update_q
    estimate_w = mean_state[1] + update_w

    return [estimate_q, estimate_w]    
 
def covariance_update(Pxx, K, Pvv):
    res = np.matmul(K, Pvv)
    res = np.matmul(res, np.transpose(K))
    return Pxx - res
