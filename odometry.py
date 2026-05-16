import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
import json
from utilities.readResults import displayResults

def data_set_setup(sequence, dir=None) -> tuple:
    """
    return:  (left_images_list, right_images_list, P0, P1, groundTruth, times)
    """
    if dir is None:
        dir = f'./kittiDataSet/sequences'
    
    seq_dir = f'{dir}/sequences/{sequence}/'
    poses_dir = f'{dir}/poses/{sequence}.txt'
    poses = pd.read_csv(poses_dir, delimiter=' ', header=None)

    # Get names of files to iterate through
    left_image_files = sorted(f for f in os.listdir(seq_dir + 'image_0') if not f.startswith('._'))
    right_image_files = sorted(f for f in os.listdir(seq_dir + 'image_1') if not f.startswith('._'))

     # Get calibration details for scene
    calib = pd.read_csv(seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
    P0 = np.array(calib.loc['P0:']).reshape((3,4)) # left 
    P1 = np.array(calib.loc['P1:']).reshape((3,4)) # right

    # Get times and ground truth poses
    times = np.array(pd.read_csv(seq_dir + 'times.txt', delimiter=' ', header=None))
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

    # get first images --- Currently not used
    first_image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[0], cv2.IMREAD_UNCHANGED)
    first_image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[0], cv2.IMREAD_UNCHANGED)
    imheight = first_image_left.shape[0]
    imwidth = first_image_left.shape[1]

    return (left_image_files, right_image_files, P0, P1, gt, times)


def algorithm_1(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 1 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo SGBM, SIFT, a BF matcher, and no matches pruning 
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''
    
    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    #f = k_left[0][0]            # focal length of x axis for left camera
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        sad_window = 6
        num_disparities = sad_window * 16
        block_size = 11
            
        
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 1 * block_size ** 2,
                                        P2 = 32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            

        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        
        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        

        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)

        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top
            
        # Estimate motion between sequential images of the left camera
        
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]

        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}')
        
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_2(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 2 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo BM, SIFT, a BF matcher, and no matches pruning 
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''

    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]



    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        

        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top
            
        # Estimate motion between sequential images of the left camera
        
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_3(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 3 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo BM, SIFT, a BF matcher, uses the 100 matches with the closest match
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''
    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]



    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        

        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # Only take the top 100 matches
        if(len(matches)>100):
            matches = matches[:100]
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_4(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 4 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo BM, SIFT, a BF matcher, and Lowe's ratio test for matches pruning
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''

    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        # matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.3*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_5(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 5 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo SGBM, SIFT, a BF matcher, and Lowe's ratio test for matches pruning
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''

    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
         
            
        matcher = cv2.StereoSGBM_create(numDisparities=80,
                                        minDisparity=3,
                                        blockSize=21,
                                        P1 = 8 * 1 * 21 ** 2,
                                        P2 = 32 * 1 * 21 ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            

        disp = matcher.compute(image_left, image_right).astype(np.float32)/16            
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        # matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.3*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_6(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 6 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo BM, ORB, a BF matcher, and Lowe's ratio test for matches pruning
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''

    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.ORB_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        #matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.65*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_7(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 7 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo BM, ORB, a BF matcher, and Lowe's ratio test for matches pruning
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''

    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        
        matcher = cv2.StereoSGBM_create(numDisparities=80,
                                        minDisparity=3,
                                        blockSize=21,
                                        P1 = 8 * 1 * 21 ** 2,
                                        P2 = 32 * 1 * 21 ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.ORB_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        #matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.65*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def algorithm_8(start_pose:int = None, end_pose:int = None, live_plot = 1, gtInt:int = None):
    '''
    Algorithm 8 used to perform visual odometry on a sequence from the KITTI visual odometry dataset.
    Uses Stereo BM, ORB, a FLANN matcher, and Lowe's ratio test for matches pruning
    
    Optional Arguments:
    start_pose -- (int) starting frame number
    end_pose -- (int) ending frame number
    live_plot -- (int)  1 or 0.  If 1 a plot will be displayed that updates as the VO is computed
    gtInt -- (int) frequency to inject ground truth into the algorithm.  Input a value between 0 and 0.1 for no ground truth injection
    
    Returns:
    trajectory -- (list) Array of shape Nx3x4 of estimated poses of vehicle for each computed frame.
    mean_time -- (float) Average computation time per frame
    total_time -- (float) Total computation time
    '''

    if(end_pose == None or end_pose>len(left_image_files)):
        end_pose = len(left_image_files)
    if(start_pose == None or start_pose<0):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'{dir}/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.ORB_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        FLANN_INDEX_KDTREE = 0
        index_parameter = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_parameter = dict(checks = 20)
        matcher = cv2.FlannBasedMatcher(index_parameter, search_parameter)
        matches = matcher.knnMatch(np.asarray(des0, np.float32), np.asarray(des1, np.float32), k=2)
        #matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.65*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
        if gtInt > 0 and (i+1)%gtInt==0:
            T_tot = gt[start_pose+i+1]
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        if live_plot:
            plt.plot(xs, ys, zs, c='r')
            plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    return trajectory, mean_time, total_time


def save_results(results, gt, mean_time, total_time, abserror, relerror, angerror, alg_des, start_pose, end_pose, path):
    '''
    Saves the results from an algorithm to a JSON file so that they can be loaded and viewed later
    
    Arguments:
        results -- (array) Nx3x4 of transformation matrices that are the output of the VO algo
        gt -- (array) Nx3x4 of ground truth transformation matrices 
        mean_time -- (float) Average image processing time
        total_time -- (float) Total execution time of an algorithm
        abserror -- (list) List of floats for the absolute error of each frame of the algorithm
        relerror -- (list) List of floats for the relative error of each frame of the algorithm
        angerror -- (list) List of angle errors of each frame of the algorithm
        alg_des -- (string) String describing the algorithm
        start_pose -- (int) The starting position of the algorithm
        end_pose -- (int) The ending position of the algorithm
        path -- (string) The path to save the image as
    
    Returns:
        Nothing
    '''

    results_writable = []
    for i in range(len(results)):
        results_writable.append(list(np.ravel(results[i])))

    gt_writable = []
    for i in range(len(gt)):
        gt_writable.append(list(np.ravel(gt[i]))) 

    abserror_writable = []
    for i in range(len(abserror)):
        abserror_writable.append(list(np.ravel(abserror[i]))) 

    relerror_writable = []
    for i in range(len(relerror)):
        relerror_writable.append(list(np.ravel(relerror[i]))) 
        
    angerror_writable = []
    for i in range(len(angerror)):
        angerror_writable.append(list(np.ravel(angerror[i]))) 
    
    data_to_write = {
        "algorithm description": alg_des,
        "odometry" : results_writable,
        "ground truth": gt_writable,
        "absolute error": abserror_writable,
        "relative error": relerror_writable,
        "angular heading error": angerror_writable,
        "mean time" : mean_time,
        "total time" : total_time,  
        "start pose" : start_pose,
        "end pose"   : end_pose,
    }

    with open(path, "w") as outfile:
        json.dump(data_to_write, outfile)


def compute_error(gt,computed_trajectory,start_pose):
    """compute error between ground truth and computed trajectory, 
    Args:
        gt: ground truth data
        computed_trajectory: estimated trajectory from VO
        start_pose (int): the frame that we began the VO at
    Returns:
        abserror (list): the distances between the actual and the estimated locations
        relerror (list): the differences between the distance between the current and next estimated points and the distance between the current and next actual points
        angerror (list): the differences between the angle formed by the previous, current, and next estimated points and the angle formed by the previous, current, and next actual points
    """
    #if no correct starting frame, make it 0
    if(start_pose == None or start_pose<0):
        start_pose = 0
    #get relevant ground truth data
    gt = gt[start_pose:,:,3]
    # get relevant estimated trajectory
    computed_trajectory = computed_trajectory[:,:,3]
    #initialize error lists
    abserror = []
    relerror = []
    angerror = []

    #calculation of absolute error
    for i in range(len(computed_trajectory)):
        abserror.append(abs(np.linalg.norm(gt[i]-computed_trajectory[i])))

    #calculation of relative errors
    for j in range(1,len(computed_trajectory)-1):
        #relativeDistanceError = norm( |distanceBetweenTwoGroundTruthPoints| - |distanceBetweenTwoEstimatedPoints| )
        relerror.append(abs(np.linalg.norm(abs(np.linalg.norm(gt[j]-gt[j+1]))-abs(np.linalg.norm(computed_trajectory[j]-computed_trajectory[j+1])))))
        
        #relativeAngleError = |angleBetweenThreeGroundTruthPoints| - |angleBetweenThreeEstimatedPoints|
        ba1 = gt[j-1] - gt[j]
        bc1 = gt[j+1] - gt[j]
        cosine_angle1 = np.dot(ba1, bc1) / (np.linalg.norm(ba1) * np.linalg.norm(bc1))
        angle1 = np.arccos(cosine_angle1)

        ba2 = computed_trajectory[j-1] - computed_trajectory[j]
        bc2 = computed_trajectory[j+1] - computed_trajectory[j]
        cosine_angle2 = np.dot(ba2, bc2) / (np.linalg.norm(ba2) * np.linalg.norm(bc2))
        angle2 = np.arccos(cosine_angle2)

        angerror.append(abs(np.degrees(angle1)-np.degrees(angle2)))
    return abserror,relerror,angerror


if __name__ == "__main__":
    
    start_time = datetime.now()
    frame_rate = 10 #Hz
    sequence = "00"
    # set the directory for where the sequence data is stored.
    dir = f"/Volumes/SSK Media/visual_odometry/dataset" 
    
    left_image_files, right_image_files, P0, P1, gt, times = data_set_setup(sequence, dir)
    # Setup plot that will be used on each iteration of code
    fig = plt.figure(figsize=(7, 7))
    plt.title("Trajectory")
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = gt[:, 0, 3]
    ys = gt[:, 1, 3]
    zs = gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='b')

    # Choose Algorithm
    description_1 = "Algorithm 1: SGBM + SIFT  + BF "
    description_2 = "Algorithm 2: BM + SIFT + BF "
    description_3 = "Algorithm 3: BM + SIFT + BF + Filter: Top 100 Matches "
    description_4 = "Algorithm 4: BM + SIFT + BF + Filter: Lowe Ratio Test "
    description_5 = "Algorithm 5: SGBM + SIFT + BF + Filter: Lowe Ratio Test "
    description_6 = "Algorithm 6: BM + ORB + BF + Filter: Lowe Ratio Test"
    description_7 = "Algorithm 7: SGBM + ORB + BF + Filter: Lowe Ratio Test"
    description_8 = "Algorithm 8: BM + ORD + FLANN + Filter: Lowe Ratio Test"
    

    p1 = f"{dir}/results/algorithm_1/algorithm_1.json" 
    p2 = f"{dir}/results/algorithm_2/algorithm_2.json"
    p3 = f"{dir}/results/algorithm_3/algorithm_3.json"
    p4 = f"{dir}/results/algorithm_4/algorithm_4.json"
    p5 = f"{dir}/results/algorithm_5/algorithm_5.json"
    p6 = f"{dir}/results/algorithm_6/algorithm_6.json"
    p7 = f"{dir}/results/algorithm_7/algorithm_7.json"
    p8 = f"{dir}/results/algorithm_8/algorithm_8.json"

    #user interface initialization and processes
    print("Menu: ")
    print("STEREO MATCHER + FEATURE DETECTOR + FEATURE MATCHER + FEATURE MATCH FILTER")
    print(description_1)
    print(description_2)
    print(description_3)
    print(description_4)
    print(description_5)
    print(description_6)
    print(description_7)
    print(description_8)
    print("Enter -1 to Automatically collect data from All algorithms")

    alg_num = input("Enter Algorithm Number: ")

    alg_num = int(alg_num)


    auto = 0
    match alg_num:
        case -1:
            auto = 1
            alg_des = 'AUTO-ALL'
        case 1:
            alg = algorithm_1
            alg_des = description_1
            path = f"{dir}/results/algorithm_1/algorithm_1.json"
        case 2:
            alg = algorithm_2
            alg_des = description_2
            path = f"{dir}/results/algorithm_2/algorithm_2.json"
        case 3:
            alg = algorithm_3
            alg_des = description_3
            path = f"{dir}/results/algorithm_3/algorithm_3.json"
        case 4:
            alg = algorithm_4
            alg_des = description_4
            path = f"{dir}/results/algorithm_4/algorithm_4.json"
        case 5:
            alg = algorithm_5
            alg_des = description_5
            path = f"{dir}/results/algorithm_5/algorithm_5.json"
        case 6:
            alg = algorithm_6
            alg_des = description_6
            path = f"{dir}/results/algorithm_6/algorithm_6.json"
        case 7:
            alg = algorithm_7
            alg_des = description_7
            path = f"{dir}/results/algorithm_7/algorithm_7.json"
        case 8:
            alg = algorithm_8
            alg_des = description_8
            path = f"{dir}/results/algorithm_8/algorithm_8.json"
        case default:
            alg = algorithm_1
            alg_des = description_1
            path = f"{dir}/results/algorithm_1/algorithm_1.json"
    
    print()
    print("CHOSEN:")
    print(alg_des)
    #if chose to not run all, allow for overriding path location

    if alg_num != -1:
        temp = input('Temporary Run [0/1] - Will override path location to save data in separate temporary folder under algorithm folder: ')
        temp = int(temp)
        if temp:
            path = path[:35] + '/temp/temp.json'
    #ask to show live plot
    live_plot = input('Show live plot [0/1]: ')
    live_plot = int(live_plot)

    #ask to save data to file
    save_json_data = input('Save JSON Data: [0/1] - Needed ON to display/save end plots + summary: ')
    save_json_data = int(save_json_data)

    #ask what to save and show if saving to file
    if save_json_data:
        show_plots = input('Show All Plots: [0/1]: ')
        show_plots = int(show_plots)
        
        save_plots = input('Save All Plots [0/1]: ')
        save_plots = int(save_plots)
    else:
        show_plots = 0
        save_plots = 0

    #enter beginning and ending position
    start_pose = input("Enter Start Frame: ")
    start_pose = int(start_pose)
    end_pose = input("Enter End Frame: ")
    end_pose = int(end_pose)

    #enter frequency to inject ground truth
    gtInt = input("Enter frequency (in seconds) to inject ground truth data (Enter value < 0.1 if never): ")
    gtInt = int(round(frame_rate*float(gtInt)))
    
    dummy = input('dummy')
    
    
    algs = [algorithm_1, algorithm_2, algorithm_3, algorithm_4, algorithm_5, algorithm_6, algorithm_7, algorithm_8]
    alg_descrps = [description_1, description_2, description_3, description_4, description_5, description_6, description_7, description_8]
    alg_paths = [p1,p2,p3,p4,p5,p6,p7,p8]

    # automatically loops through all algorithms for data collection
    if auto:
        print('BEGINNING AUTOMATIC DATA COLLECTION OF ALL ALGORITHMS')
        print('--WILL NOT DISPLAY ANY PLOTS, BUT WILL SAVE ALL DATA [JSON, PLOTS]--')
        for i in range(len(algs)):
            alg = algs[i]
            alg_des = alg_descrps[i]
            path = alg_paths[i] 
            
            # Run algorithm
            # added third argument: 0,1 - LIVE PLOTTING - shows real time plot. DEFAULT: 1 [ON]
            computed_trajectory, mean_time, total_time = alg(start_pose, end_pose, live_plot = 0, gtInt = 0.001)
            
            # Compute Error 
            abserror,relerror,angerror = compute_error(gt, computed_trajectory,start_pose)

            # Save/Overwrite result data
            save_results(computed_trajectory, gt, mean_time, total_time, abserror, relerror, angerror, alg_des, start_pose, end_pose, path)

            # display/save figs + summary
            # first argument: path
            # second argument: 0,1 - DISPLAY - shows all plots. DEFAULT: 1 [ON]
            # thirds argument: 0,1 - SAVE - saves all images + summary in respective folder locations. DEFAULT: 0 [OFF]
            # NOTE: SAVE ON WILL OVERWRITE EXISITING FILES WITH SAME NAMES IN DESGINATED FOLDERS

            displayResults(path, display = 0 , save = 1, temp=0)
        
        print('AUTOMATIC DATA COLLECTION FINISHED')
    else:
        computed_trajectory, mean_time, total_time = alg(start_pose, end_pose, live_plot, gtInt)
        abserror,relerror,angerror = compute_error(gt, computed_trajectory,start_pose)
        plt.waitforbuttonpress()
        if save_json_data:
            plt.close()
            save_results(computed_trajectory, gt, mean_time, total_time, abserror, relerror, angerror, alg_des, start_pose, end_pose, path)
            displayResults(path, show_plots, save_plots, temp)
    
    
    
    print('Finished')


    
    

    
