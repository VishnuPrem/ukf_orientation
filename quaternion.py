
import numpy as np
import math

class Quaternion:
    
    def __init__(self, w, x, y, z, multiply = False):
        self.q = np.array([w,x,y,z])
        if not multiply:
            self.normalise()
        
    def __repr__(self):
        return "[ " + str(self.q[0]) + ", " + str(self.q[1]) + ", " + str(self.q[2]) + ", " + str(self.q[3]) + "]"
        
    def __str__(self):
        return "[ " + str(self.q[0]) + ", " + str(self.q[1]) + ", " + str(self.q[2]) + ", " + str(self.q[3]) + "]"
        
    def __mul__(self, other):      
        w_ = self.q[0] * other.q[0] - self.q[1] * other.q[1] - self.q[2] * other.q[2] - self.q[3] * other.q[3]      
        u0_ = self.q[0]
        u_ = self.q[1:]
        v0_ = other.q[0]
        v_ = other.q[1:]      
        xyz_ = u0_*v_ + v0_*u_ + np.cross(u_, v_)      
        return Quaternion(w_, xyz_[0], xyz_[1], xyz_[2], True)
        
    def norm(self):
        return np.sqrt(self.q[0]*self.q[0] + self.q[1]*self.q[1] + self.q[2]*self.q[2] + self.q[3]*self.q[3])
        
    def inverse(self):
        self.q[1] *= -1
        self.q[2] *= -1
        self.q[3] *= -1
    
    def normalise(self):
        norm = 0
        for val in self.q:
            norm += val*val
        if norm == 0:
            return
        self.q = self.q / np.sqrt(norm)

    def euler_angles(self):
        r = math.atan2(2*(self.q[0]*self.q[1]+self.q[2]*self.q[3]), \
                1 - 2*(self.q[1]**2 + self.q[2]**2))
        p = math.asin(2*(self.q[0]*self.q[2] - self.q[3]*self.q[1]))*-1
        y = math.atan2(2*(self.q[0]*self.q[3]+self.q[1]*self.q[2]), \
                1 - 2*(self.q[2]**2 + self.q[3]**2))*-1
        if r > 0:
            r= r - 3
        elif r < 0:
            r = r + 3
            
        if y > 2.5 or y < -2.5:
            y = 0.0
            
        N = 100
        r = np.convolve(r,np.ones((N,))/N,mode = 'same')
        p = np.convolve(p,np.ones((N,))/N,mode = 'same')
        y = np.convolve(y,np.ones((N,))/N,mode = 'same')
        
        return np.array([r, p, y])
    
    def from_rotm(self, R):
       theta = math.acos((np.trace(R)-1)/2)
       omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
       omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
       self.q[0] = math.cos(theta/2)
       self.q[1:4] = omega*math.sin(theta/2)
       self.normalize()
       
    @staticmethod
    def average_quats(quats, q_mean): 
        
        # q_mean = Quaternion(random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1),)
        # q_mean.normalise()
        
        itr = 0
        while True: 
                   
            n = len(quats)
            e_list = np.zeros((4, n))
            q_mean.inverse()
            
            # compute e for all qi
            for i, qi in enumerate(quats):
                e = qi*q_mean
                e_list[:,i] = np.array([e.q[0], e.q[1], e.q[2], e.q[3]])  
                #print(e_list)
            
            # convert to angle axis representation
            angle = 2 * np.arccos(e_list[0,:])
            axis_x = e_list[1,:] / np.sqrt(1 - e_list[0,:]*e_list[0,:])
            axis_y = e_list[2,:] / np.sqrt(1 - e_list[0,:]*e_list[0,:]) 
            axis_z = e_list[3,:] / np.sqrt(1 - e_list[0,:]*e_list[0,:])
            
            axis_x *= angle
            axis_y *= angle
            axis_z *= angle
            
            # find mean of angle axis
            mean_axis_x = np.sum(axis_x)
            mean_axis_y = np.sum(axis_y)
            mean_axis_z = np.sum(axis_z)
            
            e_vector = [mean_axis_x, mean_axis_y, mean_axis_z]
            
            mean_axis_x /= n
            mean_axis_y /= n
            mean_axis_z /= n
            
            # convert to quaternion
            angle = np.sqrt(mean_axis_x*mean_axis_x + mean_axis_y*mean_axis_y + mean_axis_z*mean_axis_z)
            mean_axis_x /= angle
            mean_axis_y /= angle
            mean_axis_z /= angle
            
            e_w = np.cos(angle/2)
            e_x = np.sin(angle/2) * mean_axis_x
            e_y = np.sin(angle/2) * mean_axis_y
            e_z = np.sin(angle/2) * mean_axis_z
            
            e_mean = Quaternion(e_w, e_x, e_y, e_z)            
            q_mean.inverse()
            
            # compute new mean
            q_mean = e_mean * q_mean
            
            e_vec_mag = np.sqrt(e_vector[0]*e_vector[0] + e_vector[1]*e_vector[1] + e_vector[2]*e_vector[2])
            
            itr+=1
            
            if e_vec_mag < 0.01 or itr > 10000:
                return q_mean
        
    @staticmethod
    def angleaxis_to_quat(angle, axis):
        # axis: 3x1
        # angle: scalar
        
        q0 = np.cos(angle/2)
        
        magnitude = math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
        
        if magnitude == 0:
            return Quaternion(1,0,0,0)
        
        axis = axis/magnitude
        q123 = np.sin(angle/2) * axis
        q = Quaternion(q0.item(), q123[0], q123[1], q123[2])
        q.normalise()
        return q
    
    @staticmethod
    def magnaxis_to_quat(axis):
        # axis: 3x1
        angle = math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
        if angle == 0:
            return Quaternion(1, 0, 0, 0)
        
        axis = axis/angle
        return Quaternion.angleaxis_to_quat(angle, axis)
    
    def quat_to_magnaxis(self):
        angle = np.arccos(self.q[0]) * 2
        axis_x = self.q[1] / np.sqrt(1 - self.q[0] * self.q[0])
        axis_y = self.q[2] / np.sqrt(1 - self.q[0] * self.q[0]) 
        axis_z = self.q[3] / np.sqrt(1 - self.q[0] * self.q[0])      
        axis_x *= angle
        axis_y *= angle
        axis_z *= angle
        magnaxis = np.array([axis_x, axis_y, axis_z])
        return magnaxis

# q1 = Quaternion(0.5,-1,-1,-1)
# q1.normalise()
# print(q1)

# q2 = Quaternion(0.7,-2,-1,-3)
# q2.normalise()
# print(q2)


# q3 = Quaternion(1,2,2,2)
# q3.normalise()
# print(q3)

# q4 = Quaternion(10,11,12,13)
# print(q1)

# q5 = Quaternion(14,13,12,11)
# print(q2)


# q6 = Quaternion(-1,-2,2,2)
# print(q6)

# q_mean = Quaternion(1,0,0,0)

# q_list = [q1, q2]
# print(Quaternion.average_quats(q_list, q_mean))

