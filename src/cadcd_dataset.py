
import os
import glob
import numpy as np
from collections import namedtuple
from termcolor import cprint
import time
import datetime
from utils_numpy_filter import NUMPYIEKF as IEKF
from scipy.interpolate import interp1d
import math
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CADCDataset():

    NovatelRTKPacket = namedtuple('NovatelRTKPacket', 
                                      'latitude, longitude, altitude, undulation, '
                                      'latitude_std, longitude_std, altitude_std, '
                                      'roll, pitch, azimuth, '
                                      'roll_std, pitch_std, azimuth_std, '
                                      'ins_status, position_type')

    NovatelRTKData = namedtuple('NovatelRTKData', 'packet, T_w_imu')
    min_seq_dim = 30 * 100
    def __init__(self):
        self.path_data_base = "/home/tariqul/repos/cadc_data/cadcd_data"
        



    def read_data(self):
    # Define the NovatelRTKPacket namedtuple
        
        print("Start read_data")
        t_tot = 0  # sum of times for the all dataset
        date_dirs = sorted(os.listdir(self.path_data_base))
        for n_iter, date_dir in enumerate(date_dirs):
                # get access to each sequence
                path1 = os.path.join(self.path_data_base, date_dir)
                if not os.path.isdir(path1):
                    continue
                date_dirs2 = sorted(os.listdir(path1))
                for date_dir2 in date_dirs2:
                    path2 = os.path.join(path1, date_dir2)
                    if not os.path.isdir(path2):
                        continue
                    novatel_rtk_files = sorted(glob.glob(os.path.join(path2, 'novatel_rtk', 'data', '*.txt')))    
                    novatel_rtk = CADCDataset.load_novatel_rtk_packets_and_poses(novatel_rtk_files)
                    print("\n Sequence name : " + date_dir2)
                    if len(novatel_rtk) < CADCDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
                        cprint("Dataset is too short ({:.2f} s)".format(len(novatel_rtk) / 100), 'yellow')
                        continue

                    t_novatel_rtk = CADCDataset.load_timestamps(path2)
                    lat_novatel_rtk = np.zeros(len(novatel_rtk))
                    lon_novatel_rtk = np.zeros(len(novatel_rtk))
                    alt_novatel_rtk = np.zeros(len(novatel_rtk))
                    roll_novatel_rtk = np.zeros(len(novatel_rtk))
                    pitch_novatel_rtk = np.zeros(len(novatel_rtk))
                    yaw_novatel_rtk = np.zeros(len(novatel_rtk))
                    roll_gt = np.zeros(len(novatel_rtk))
                    pitch_gt = np.zeros(len(novatel_rtk))
                    yaw_gt = np.zeros(len(novatel_rtk))
                    p_gt = np.zeros((len(novatel_rtk), 3))
                    k_max = len(novatel_rtk)
                    for k in range(k_max):
                        novatel_rtk_k = novatel_rtk[k]
                        t_novatel_rtk[k] = 3600 * t_novatel_rtk[k].hour + 60 * t_novatel_rtk[k].minute + t_novatel_rtk[k].second + t_novatel_rtk[k].microsecond / 1e6
                        lat_novatel_rtk[k] = novatel_rtk_k[0].latitude
                        lon_novatel_rtk[k] = novatel_rtk_k[0].longitude
                        alt_novatel_rtk[k] = novatel_rtk_k[0].altitude
                        
                        # convert from RFU to FLU frame and taking  east as zero yaw.Not taking negative pitch as the 
                        #Rotation matrix is calculated considering the negative pitch
                        roll_novatel_rtk[k] = np.radians(novatel_rtk_k[0].roll)
                        pitch_novatel_rtk[k] = np.radians(novatel_rtk_k[0].pitch)
                        yaw_novatel_rtk[k] = np.radians(-1 * novatel_rtk_k[0].azimuth) 

                        p_gt[k] = novatel_rtk_k[1][:3, 3]
                        Rot_gt_k = novatel_rtk_k[1][:3, :3]
                        roll_gt[k], pitch_gt[k], yaw_gt[k] = CADCDataset.to_rpy_updated(Rot_gt_k)
                        print('yaw_gt : {}'.format(yaw_gt))
                    
                    vel_novatel = CADCDataset.load_novatel_velocities(path2)
                    print('novatel velocitties loaded')
                    t_novatel = CADCDataset.load_timestamps(path2, novatel=True)
                    # t_novatel =  np.zeros(len(t_novatel_datetime))
                    for l in range(len(t_novatel)):
                        t_novatel[l] = 3600 * t_novatel[l].hour + 60 * t_novatel[l].minute + t_novatel[l].second + t_novatel[l].microsecond / 1e6
                    print('Novatel time loaded')
                    v_gt_interpolated = CADCDataset.interpolate_velocities(t_novatel, vel_novatel, t_novatel_rtk)
                    print ('velocity inerpolated)')
                    t0 = t_novatel_rtk[0]
                    t = np.array(t_novatel_rtk) - t0
                    if np.max(t[:-1] - t[1:]) > 0.1:
                        cprint(date_dir2 + " has time problem", 'yellow')
                    ang_gt = np.zeros((roll_gt.shape[0], 3))
                    ang_gt[:, 0] = roll_gt
                    ang_gt[:, 1] = pitch_gt
                    ang_gt[:, 2] = yaw_gt

                    #imu data in RFU frame
                    imu_rfu = CADCDataset.load_novatel_imu(path2)
                    
                    # convert from numpy
                    t = torch.from_numpy(t)
                    p_gt = torch.from_numpy(p_gt)
                    v_gt = torch.from_numpy(v_gt_interpolated)
                    ang_gt = torch.from_numpy(ang_gt)
                    

    @staticmethod
    def load_novatel_rtk_packets_and_poses(novatel_rtk_files):
            """Generator to read ground truth data.
               Poses are given in an East-North-Up coordinate system
               whose origin is the first GPS position.
            """
            # Scale for Mercator projection (from first lat value)
            scale = None
            # Origin of the global coordinate system (first GPS position)
            origin = None

            novatel_rtk = []

            for filename in novatel_rtk_files:
                with open(filename, 'r') as f:
                    for line in f.readlines():
                        line = line.split()
                        # Last five entries are flags and counts
                        line[:-2] = [float(x) for x in line[:-2]]
                        line[-2:] = [int(float(x)) for x in line[-2:]]

                        packet = CADCDataset.NovatelRTKPacket(*line)

                        if scale is None:
                            scale = np.cos(packet.latitude * np.pi / 180.)

                        R, t = CADCDataset.pose_from_novatel_rtk_packet(packet, scale)

                        if origin is None:
                            origin = t

                        T_w_imu = CADCDataset.transform_from_rot_trans(R, t - origin)
                        novatel_rtk.append(CADCDataset.NovatelRTKData(packet, T_w_imu))
            return novatel_rtk

    @staticmethod
    def pose_from_novatel_rtk_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an novatel_rtk packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.longitude * np.pi * er / 180.
        ty = scale * er * np.log(np.tan((90. + packet.latitude) * np.pi / 360.))
        tz = packet.altitude    
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        roll_rad = np.radians(packet.roll)
        pitch_rad = np.radians(packet.pitch)
        yaw_rad = np.radians(-1 * packet.azimuth)
        R_z_90 = np.array([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])

        #computed local level to body frame rotation matrix with Rx, Rx, Rz.
        # This is an extrinsic rotation
        Rx = CADCDataset.rotx(pitch_rad)
        Ry = CADCDataset.roty(roll_rad)
        Rz = CADCDataset.rotz(yaw_rad)
        R = (Ry.dot(Rx.dot(Rz)))
        [roll, pitch, yaw] = CADCDataset.to_rpy_updated(R)

        # c_phi = math.cos(roll_rad)  
        # s_phi = math.sin(roll_rad)
        # c_theta = math.cos(pitch_rad)
        # s_theta = math.sin(pitch_rad)
        # c_psi = math.cos(yaw_rad)
        # s_psi = math.sin(yaw_rad)
        
        
        # R_alt = np.array([[c_psi * c_phi - s_psi * s_theta * s_phi, -s_psi * c_theta, c_psi * s_phi + s_psi * s_theta * c_phi],
        #               [s_psi * c_phi + c_psi * s_theta * s_phi, c_psi * c_theta, s_psi * s_phi - c_psi * s_theta * c_phi],
        #               [-c_theta * s_phi, s_theta, c_theta * c_phi]]).T
        return R, t

        
    #changed the rotx, roty and rotz as per novatel definition    
    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_timestamps(data_path, novatel = False):
        """Load timestamps from file."""
        if not novatel:
            timestamp_file = os.path.join(data_path, 'novatel_rtk', 'timestamps.txt')
        else:
            timestamp_file = os.path.join(data_path, 'novatel', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps
    
    @staticmethod
    def load_novatel_velocities(data_path):
        """
        Load velocity data (north, east, up) from Novatel files.
    
        novatel_files: List of file paths to Novatel data files.
        Returns:
            velocities: Array of shape (N, 3) where N is the number of data points.
                        Each row contains [north_velocity, east_velocity, up_velocity].
        """
        velocities = []
        novatel_files = sorted(glob.glob(os.path.join(data_path, 'novatel', 'data', '*.txt')))
        for file in novatel_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    data = line.split()  # Adjust this based on the delimiter (space, comma, etc.)
                    
                    # Assuming the format is consistent, extract the velocity fields:
                    north_velocity = float(data[17])  # North velocity
                    east_velocity = float(data[18])   # East velocity
                    up_velocity = float(data[19])     # Up velocity
    
                    # Append to velocities list
                    velocities.append([north_velocity, east_velocity, up_velocity])
        
        return np.array(velocities)
    
    @staticmethod
    def load_novatel_imu(data_path):
        imu_data = []
        novatel_files = sorted(glob.glob(os.path.join(data_path, 'novatel_imu', 'data', '*.txt')))
        for file in novatel_files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    data = line.split()  # Adjust this based on the delimiter (space, comma, etc.)
                    
                    # Assuming the format is consistent, extract the velocity fields:
                    pitch_rate = float(data[0])*100  
                    roll_rate = float(data[1])*100   
                    yaw_rate = float(data[2])*100    
                    x_accel = float(data[3])*100  
                    y_accel = float(data[4])*100
                    z_accel = float(data[5])*100
    
                    # Append to velocities list
                    imu_data.append([pitch_rate, roll_rate, yaw_rate, x_accel, y_accel, z_accel])
        return np.array(imu_data)

    @staticmethod
    def interpolate_velocities(novatel_timestamps, velocities_novatel, rtk_timestamps):
        """
        Interpolate Novatel velocities (vn, ve, vu) to align with RTK timestamps.
        """
        velocities_rtk = np.zeros((len(rtk_timestamps), 3))
        
        for i in range(3):  # Interpolate for vn, ve, vu
            interp_func = interp1d(novatel_timestamps, velocities_novatel[:, i], fill_value="extrapolate")
            velocities_rtk[:, i] = interp_func(rtk_timestamps)
        
        return velocities_rtk
    
    @staticmethod
    def to_rpy_updated(Rot):

        pitch = np.arctan2(Rot[1, 2], np.sqrt(Rot[1, 0]**2 + Rot[1, 1]**2))

        if np.isclose(pitch, np.pi / 2.):
            yaw = 0. 
            roll = np.arctan2(Rot[2, 0], -Rot[2, 1])
        elif np.isclose(pitch, -np.pi / 2.):
            yaw = 0.
            roll = np.arctan2(Rot[2, 0], Rot[2, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(-Rot[1, 0] * sec_pitch,
                             Rot[1, 1] * sec_pitch)
            roll = np.arctan2(-Rot[0, 2] * sec_pitch,  
                              Rot[2, 2] * sec_pitch)
        return roll, pitch, yaw
    
    

    

if __name__ == '__main__':
  dataset = CADCDataset()
  dataset.read_data()