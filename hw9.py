import cv2 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math 
import random 
from copy import deepcopy

#create img for corresponding points
def corresponding_point_img(img1, img2, points1, points2):
    
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    #create side by side image 
    new_img = np.hstack((img1, img2))
    
    #get offset for the second image 
    offset = img1.shape[1]
    
    #draw all the points in the image 
    for pt1, pt2 in zip(points1, points2):
        #draw point in img1
        cv2.circle(new_img, (int(pt1[0]), int(pt1[1])), 4, 
                   (255,0,0), -1)
        #draw point in img2
        cv2.circle(new_img, (int(pt2[0]+offset), int(pt2[1])),
                   4, (255,0,0), -1)
        #draw line between points
        cv2.line(new_img,(int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]+offset), int(pt2[1])), (0,255,0), 2)
        
    #write the new_image 
    cv2.imwrite('1_corresponding_points.jpeg', new_img)
    return

def normalize_points(points):
    
    points = np.array(points)
    
    #get the translation for x and y 
    t_x = np.mean(points[:,0])
    t_y = np.mean(points[:,1])
    
    #get the mean of the distances of each point to the mean 
    distances = np.sqrt((points[:,0]-t_x)**2 + (points[:,1]-t_y)**2)
    mean_distance = np.mean(distances)
    
    #get the scale variable
    s = np.sqrt(2)/mean_distance
    
    #create the normalizing matrix 
    T = np.array([[s, 0,-s*t_x],[0,s,-s*t_y],[0,0,1]])
    
    #get the homogenous coordinates of points
    homo_points = real_to_homogeneous(points)
    
    #get the normalized points 
    normalize_points = []
    for point in homo_points:
        norm_point = T @ point
        norm_point = norm_point / norm_point[2]
        normalize_points.append((norm_point[0], norm_point[1]))
        
    normalize_points = np.array(normalize_points)
    
    return normalize_points, T 

def real_to_homogeneous(points):
    
    points = np.array(points)
    
    #add a 1 to the points to make homogeneous 
    homo_points = np.hstack((points,np.ones((len(points),1))))
    
    return homo_points 

def get_F_estimate(points1, points2, T1, T2):
    
    #create the A matrix 
    A_matrix = []
    for i in range(len(points1)):
        A = [points2[i][0]*points1[i][0], 
             points2[i][0]*points1[i][1], 
             points2[i][0], points2[i][1]*points1[i][0], 
             points2[i][1]*points1[i][1], 
             points2[i][1], points1[i][0], 
             points1[i][1], 1]
        A_matrix.append(A)
        
    #solve for F using SVD
    _, _ ,v_t = np.linalg.svd(np.transpose(A_matrix) @ A_matrix)
    F = np.array(v_t[-1])
    F_matrix = np.reshape(F, (3,3))
    
    #condition the F matrix to make sure rank = 2
    u, d,v_t = np.linalg.svd(F_matrix)
    d[2] = 0 #set smallest singualr value to 0 
    d = np.diag(d)
    new_F = u @ d @ v_t
    
    #now denormalize the F 
    F_final = np.transpose(T2) @ new_F @ T1
    F_final = F_final / F_final[2][2]
    
    return F_final

def get_epipoles(f_matrix):
    
    #use svd to get the left and right null vectors which 
    # are the epipoles
    u, d, v_t = np.linalg.svd(f_matrix)
    epipole = np.transpose(v_t[-1,:])
    epipole2 = u[:,-1]
    
    #normalize 
    epipole = epipole / epipole[2]
    epipole2 = epipole2 / epipole2[2]
    
    return epipole, epipole2

def get_P(f_matrix, epipole2):
    
    #get the P in canonical representation 
    P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    P = np.array(P)
    
    #get the P_prime in canonical representation 
    s = [[0,-epipole2[2],epipole2[1]],
         [epipole2[2],0,-epipole2[0]],[-epipole2[1],
                                       epipole2[0],0]]
    s = np.array(s)
    
    sF = s @ f_matrix
    P_prime = np.hstack((sF, epipole2.reshape(-1,1)))
        
    return P, P_prime

def get_world_coords(x, x_prime, P, P_prime):
    
    x.append(1)
    x_prime.append(1)

    #get the A matrix 
    A = []
    A.append(x[0]*P[2,:] - P[0,:])
    A.append(x[1]*P[2,:] - P[1,:])
    A.append(x_prime[0]*P_prime[2,:] - P_prime[0,:])
    A.append(x_prime[1]*P_prime[2,:] - P_prime[1,:])
    A = np.array(A)

    #solve using svd 
    _, _ ,v_t = np.linalg.svd(A)
    world_coord = np.array(v_t[-1])
    world_coord = world_coord / world_coord[-1]
    world_coord = [world_coord[0], world_coord[1], 
                   world_coord[2]]
    
    x.pop(1)
    x_prime.pop(1)
    
    return world_coord

def applyP(homography, world_coords):
    
    #convert to homogeneous coords 
    world_coords = np.array(world_coords)
    world_coords_homo = np.ones((world_coords.shape[0],4))
    world_coords_homo[:,:-1] = world_coords
    
    #get projected points
    proj_points = []
    for point in world_coords_homo:
        temp = homography @ point
        temp = temp / temp[-1]
        temp = (temp[0], temp[1])
        proj_points.append(temp)
    proj_points = np.array(proj_points)
    
    return proj_points

def applyH(homography, points):
    
    #convert to homogenouse 
    homo_points = [[x, y, 1] for x, y in points]
    
    #get projected points
    proj_points = []
    for point in homo_points:
        temp = homography @ point
        temp = temp / temp[-1]
        temp = (temp[0], temp[1])
        proj_points.append(temp)
    proj_points = np.array(proj_points)
    
    return proj_points


def cost_func(parameters, points1, points2):
    
    #get the varibles from the parameters
    M = parameters[:9]
    t = parameters[9:12]
    
    M = np.reshape(M, (3,3))
    t = np.reshape(t, (3,1))
    
    P_prime = np.hstack((M,t))
    
    #get world coordinates 
    world_coords = []
    for i in range(len(points1)):
        world_coords.append(parameters[12+3*i:12+3*(i+1)])
        
    P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        
    #now do reprojectoin of world coords to image plane 
    x_proj = applyP(P, world_coords)
    x_prime_proj = applyP(P_prime, world_coords)
    
    x_error = np.subtract(points1, x_proj)
    x_prime_error = np.subtract(points2, x_prime_proj)
    
    x_error_x = x_error[:,0]
    x_error_y = x_error[:,1]
    x_prime_error_x = x_prime_error[:,0]
    x_prime_error_y = x_prime_error[:,1]
    
    cost = np.hstack((x_error_x, x_error_y, x_prime_error_x, 
                      x_prime_error_y))

    return cost

def apply_homography(img, homography, output_size=None):
    
    #calculate new image boundaries 
    height, width = img.shape[:2]
    
    # Define the corner points of the input image
    corners = np.array([
        [0, 0],        
        [width, 0],    
        [width, height], 
        [0, height]      
    ], dtype=np.float32)

    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    new_corners = np.matmul(homography, corners_homogeneous.T).T
    new_corners /= new_corners[:, 2:3] 
    
    min_x = np.floor(np.min(new_corners[:, 0])).astype(int)
    max_x = np.ceil(np.max(new_corners[:, 0])).astype(int)
    min_y = np.floor(np.min(new_corners[:, 1])).astype(int)
    max_y = np.ceil(np.max(new_corners[:, 1])).astype(int)
    
    #now create new image 
    new_width = max_x - min_x
    new_height = max_y - min_y

    new_img = np.zeros((new_height, new_width, img.shape[2]), 
                       dtype=img.dtype)
    mask = np.zeros((new_height, new_width), dtype=bool)
        
    #create meshgrid
    x = np.arange(new_width)
    y = np.arange(new_height)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    #convert to homogeneous form
    homogeneous_coords = np.vstack([X_flat + min_x , 
                                    Y_flat + min_y, 
                                    np.ones(X.size)])
        
    #apply the homography
    inverse_homography = np.linalg.inv(homography)
    original_coords = np.matmul(inverse_homography, 
                                homogeneous_coords).T
    original_coords /= original_coords[:, 2:3] 
        
    #turn back into 2-d
    X_orig = np.clip(original_coords[:, 0].astype(int), 
                     0, width - 1)
    Y_orig = np.clip(original_coords[:, 1].astype(int), 
                     0, height - 1)
    
    # Update the mask where the coordinates are valid
    valid_mask = (original_coords[:, 0] >= 0) & \
    (original_coords[:, 0] < width) & \
                 (original_coords[:, 1] >= 0) & \
                     (original_coords[:, 1] < height)
    mask[Y_flat, X_flat] = valid_mask
    
    # Assign pixel values where mask is True
    new_img[mask] = img[Y_orig[valid_mask], 
                        X_orig[valid_mask]]
        
    return new_img

def get_rectficication_homography(img2, epipole2, 
                                  P, P_prime, points1, 
                                  points2):
    
    #get the shape of the image
    h, w = img2.shape

    #get the T1 matrix
    T1 = np.array([[1,0,-(w/2)],[0,1,-(h/2)],[0,0,1]])

    #get the T2 matrix
    T2 = np.array([[1,0,(w/2)],[0,1,(h/2)],[0,0,1]])
    
    theta = np.arctan(-(epipole2[1]-(h/2))/\
        (epipole2[0]-(w/2)))  # theta
    R = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta),np.cos(theta),0],[0,0,1]])
    
    #get the G matrix
    f = np.abs((epipole2[0]-(w/2))*np.cos(t)-\
        (epipole2[1]-(h/2))*np.sin(t))
    G = np.array([[1,0,0],[0,1,0],[-1/f,0,1]])

    #create H2
    H2 = T2 @ G @ R @ T1
    
    #create H1 by doing H_a @ H_0 
    M = P_prime @ np.linalg.inv(P)
    H_0 = H2 @ M

    #transform the points using H0 
    x_hat = np.zeros((len(points1), 3))
    for i, pt in enumerate(points1):
        val = H0 @ np.array([pt[0], pt[1], 1])
        x_hat[i] = val / val[-1]


    #ge the projection of x_hat    
    x_prime_hat = []
    for i in points2:
        proj = H2 @ (np.append(i, 1))
        x_prime_hat.append(proj)
    x_prime_hat = np.array(x_prime_hat)
    x_prime_hat /= x_prime_hat[:,2][:,None]

    #use linear least squares again to solve for H_a
    A_matrix = np.zeros((2*len(points1), 9))
    for i in range(len(points1)):
        A_matrix[2*i] = [x_hat[0],x_hat[1],1,0,0,0,
                         -x_prime_hat[0]*x_hat[0],
                         -x_prime_hat[0]*x_hat[1],
                         -x_prime_hat[0]]
        A_matrix[2*i+1] = [0,0,0,x_hat[0],x_hat[1],
                           1,-x_prime_hat[1]*x_hat[0],
                           -x_prime_hat[1]*x_hat[1],
                           -x_prime_hat[1]]

    #use svd to get H_a
    _ , _, v_t = np.linalg.svd(A_matrix)
    H_a = v_t[-1]
    H_a = H_a.reshape(3, 3)

    #get the H homography 
    H1 = H_a @ H_0
    H1 = H1 / H1[-1, -1]

    return H1, H2
    

def get_rectficication_homography(img2, epipole2, 
                                  P, P_prime, 
                                  points1, points2):
    
    #get the homography for right image retification - 
    # H' = T_2 @ G @ R @ T_1
    height, width, _ = img2.shape
    t1_matrix = [[1,0,-(width/2)],[0,1,-(height/2)],
                 [0,0,1]]
    t1_matrix = np.array(t1_matrix)
    
    angle = np.arctan((-epipole2[1]+(height/2))/\
        (epipole2[0]-(width/2)))
    r_matrix = [[np.cos(angle),-np.sin(angle),0],\
        [np.sin(angle), np.cos(angle),0],[0,0,1]]
    r_matrix = np.array(r_matrix)
    
    f = (epipole2[0]-(width/2))*np.cos(angle)-\
        (epipole2[1]-(height/2))*np.sin(angle)
    f = np.abs(f)
    g_matrix = [[1,0,0],[0,1,0],[-1/f,0,1]]
    
    t2_matrix = [[1,0,width/2],[0,1,height/2],[0,0,1]]
    t2_matrix = np.array(t2_matrix)
    
    H2 = t2_matrix @ g_matrix @ r_matrix @ t1_matrix
    
    #get the homography for left image retification - H = H_a @ H_0
    
    M = P_prime @ np.linalg.pinv(P)
    H_0 = H2 @ M 
    
    #use H_0 and H2 to get projected points 
    proj_points1 = applyH(H_0, points1)
    proj_points2 = applyH(H2, points2)
    proj_points1 = real_to_homogeneous(proj_points1)
    proj_points2 = real_to_homogeneous(proj_points2)
    
    #solve for H_a using  Ax-b = 0 
    abc = np.linalg.pinv(proj_points1) @ proj_points2[:,0]
    H_a = [[abc[0],abc[1],abc[2]],[0,1,0],[0,0,1]]
    H_a = np.array(H_a)
    
    H1 = H_a @ H_0
    
    return H1, H2

#for the image creation of rectified images 
def pad_bottom(img, add_pad):
    
    add_pad = int(add_pad)
    
    zero_rows = np.zeros((add_pad, img.shape[1], img.shape[2]))
    # one_row = np.zeros((1, img.shape[1], img.shape[2]))
    new_img = np.vstack((img, zero_rows))
    new_img = np.vstack((zero_rows, new_img))
    # new_img = np.vstack((one_row, new_img))
    
    return new_img

def apply_rectification_homographies(img1, img2, H1, H2, img_path):
    
    new_img1 = cv2.warpPerspective(img1, H1, (550,550))
    new_img2 = cv2.warpPerspective(img2, H2, (550,550))
    
    # new_img1 = apply_homography(img1, H1)
    # new_img2 = apply_homography(img2, H2)
    
    # new_img1_len = new_img1.shape[0]
    # new_img2_len = new_img2.shape[0]
    # len_diff = np.abs(new_img1_len - new_img2_len)
    # add_pad = len_diff // 2 
    # new_img2_temp = pad_bottom(new_img2, add_pad)
    # print(new_img2_temp.shape)
    # print(new_img1.shape)
    # # image_combined = np.concatenate((new_img1_temp, new_img2), axis=1)
    # image_combined = np.concatenate((new_img1, new_img2_temp), axis=1)
    
    cv2.imwrite(img_path+'_rec1.jpeg', new_img1)
    cv2.imwrite(img_path+'_rec2.jpeg', new_img2)
    # cv2.imwrite(img_path+'_reccombined.jpeg', image_combined)
    
    return new_img1, new_img2

def apply_canny(img,img_num):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray,(4,4))
    edge = cv2.Canny(img_gray, 300, 350)
    cv2.imwrite('1_canny_'+str(img_num)+'.jpeg', edge)
    
    return edge

def get_epipolar_canidates(row, right_points, row_tolerance=5):
    
    candidates = right_points[(right_points[:, 0] >= row - \
        row_tolerance)&(right_points[:, 0] <= row + row_tolerance)]
    
    return candidates

def get_correspondences(canny1, canny2, img1, img2, window_size=5):
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    #get the points of canny 
    left_coords = np.column_stack(np.where(canny1 > 0))
    right_coords = np.column_stack(np.where(canny2 > 0))
    
    #using ssd find the corresponding 
    matches = []
    half_window = window_size // 2
    
    for (row, col) in left_coords:
        
        #check if point is not too close to the 
        left_patch = img1[row-half_window:row+half_window+1, 
                          col-half_window:col+half_window+1]
        if left_patch.shape != (window_size, window_size):
            continue 
        
        #get epipolar canidate
        canidates = get_epipolar_canidates(row, right_coords)
        best_ssd = float('inf')
        best_match = None
        
        #look at all the canidates and see when ssd is lowest
        for (c_row, c_col) in canidates:
            
            #get canidate patch from right image 
            candidate_patch = img2[c_row-half_window:c_row+half_window+1, 
                                   c_col-half_window:c_col+half_window+1]
            if candidate_patch.shape != (window_size, window_size):
                continue 
            
            #use ssd to get score 
            ssd_score = np.sum((left_patch-candidate_patch)**2)
            if ssd_score < best_ssd:
                best_ssd = ssd_score
                best_match = (c_row, c_col)
        
        if best_match is not None:
            matches.append(((row, col), best_match))
           
    return matches

#displays the corresponding points
def display_matches(img1, img2, matches):
    
    combined_img = np.hstack((img1, img2))
    img_width = img1.shape[1]
    
    for match in matches:
        pt1 = (match[0][1], match[0][0])  #first image point
        pt2 = (match[1][1] + img_width, match[1][0])  #second image point
        color = [random.randint(0, 255) for _ in range(3)]  
        cv2.line(combined_img, pt1, pt2, color, thickness=2)
    
    cv2.imwrite('1_matching_points.jpeg', combined_img)
        
    return 

#creates the 3D Plot
def plot_3d(matches, img1_points, img2_points, P, P_prime_refined, H1, H2):
    
    P = np.array(P)
    P_prime_refined = np.array(P_prime_refined)
    
    print(P_prime_refined)
    
    #get the world coords for the actual corners
    world_coords_real = []
    for x, x_prime in zip(img1_points, img2_points):
        world_coord = get_world_coords(x,x_prime, P, P_prime_refined)
        world_coords_real.append(world_coord)
    # world_coords_real = np.array(world_coords_real)
    
    # get matches into two lists 
    left_img_points = []
    right_img_points = []
    for (left, right) in matches:
        left_img_points.append(left)
        right_img_points.append(right)
        
    #send the matching points back to orignal image from the rectified 
    left_img_points_org = applyH(np.linalg.inv(H1), left_img_points)
    right_img_points_org = applyH(np.linalg.inv(H2), right_img_points)
    
    #then get world coordinate 
    for x, x_prime in zip(left_img_points_org, right_img_points_org):
        x = list(x)
        x_prime = list(x_prime)
        world_coord = get_world_coords(x,x_prime, P, P_prime_refined)
        world_coords_real.append(world_coord)
    world_coords_real = np.array(world_coords_real)
    
    print(len(world_coords_real))
    print(world_coords_real)
    
    # img1_points = np.array(img1_points)
    # img2_points = np.array(img2_points)
    
    # left_img_points_org = np.concatenate((img1_points, 
    # left_img_points_org), axis=0)
    # right_img_points_org = np.concatenate((img2_points, 
    # right_img_points_org), axis=0)
    
    # left_img_points_org = list(left_img_points_org)
    # right_img_points_org = list(right_img_points_org)
    # img1_points = list(img1_points)
    # img2_points = list(img2_points)
    
    # #normalize the points 
    # point1_norm, T1 = normalize_points(left_img_points_org)
    # point2_norm, T2 = normalize_points(right_img_points_org)
    
    # #step 1 is to estimate the fundatmental matrix
    # f_matrix = get_F_estimate(point1_norm, point2_norm, T1, T2)
    
    # #now get the left and right epipoles
    # epipole1, epipole2 = get_epipoles(f_matrix)
    
    # #using F and epipoles find the camera matricies P and P'
    # P, P_prime = get_P(f_matrix, epipole2)
    
    # #get world coordinates uaing P 
    # world_coords = []
    # for x, x_prime in zip(left_img_points_org, right_img_points_org):
    #     x = list(x)
    #     x_prime = list(x_prime)
    #     world_coord = get_world_coords(x,x_prime, P, P_prime)
    #     world_coords.append(world_coord)
        
    # #use all world coordinates 
    # world_coords_real = np.array(world_coords_real)
    # world_coords = np.array(world_coords)
    # print(world_coords_real)
    # print(world_coords)
    # world_coords = np.concatenate((world_coords_real, 
    # world_coords), axis=0)
    # print(world_coords) 
    
    # #get the parameters for the cost function 
    # parameters = []    
    # P_prime = P_prime.ravel()
    
    # for i in P_prime:
    #     parameters.append(i)
    # for i in world_coords:
    #     parameters.append(i[0])
    #     parameters.append(i[1])
    #     parameters.append(i[2])
    
    # print(len(parameters))
    # #do LM on the new large correspondence points 
    # lm_out = least_squares(cost_func, parameters, method='lm', 
    # args=[img1_points, img2_points])
    # lm_out = lm_out.x
    
    # #recreate P from LM
    # M_refined = lm_out[0:9]
    # t_refined = lm_out[9:12]
    # P_prime_refined = np.hstack((M_refined,t_refined))
    # P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    
    # #get the world coords for the matches aquired from canny
    # world_coords_match = []
    # for x, x_prime in  zip(img1_points, img2_points):
    #     world_coord = get_world_coords(x,x_prime, P, P_prime_refined)
    #     world_coords_match.append(world_coord)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    #get 3D points from world coordinates
    x_world = world_coords_real[0, :]
    y_world = world_coords_real[1, :]
    z_world = world_coords_real[2, :]

    #plot 3D points
    ax.scatter(x_world, y_world, z_world)
    ax.set_title(f'3D Plot for Projective Stereo Reconstruction', 
                 fontsize=15)
    ax.set_xlabel('X (world)', fontsize=12)
    ax.set_ylabel('Y (world)', fontsize=12)
    ax.set_zlabel('Z (world)', fontsize=12)
    plt.show()
    
    return 
    
def main():
    
    #read in the images and get the points
    img1 = cv2.imread('img1.jpeg')
    img2 = cv2.imread('img2.jpeg')
    
    img1_points = [[36,168],[55,226],[218,311],[216,362],[437,114],
                   [417,165],[277,37],[177,103],[174,124],[229,163],
                   [227,186],[353,119],[350,140],[296,68]]
    img2_points = [[21,230],[50,285],[265,348],[265,397],[424,82],
                   [408,142],[236,32],[158,125],[157,146],[233,181],
                   [235,204],[346,107],[346,128],[271,62]]
    
    #create image with corresponing points
    corresponding_point_img(img1, img2, img1_points, img2_points)
    
    #normalize the points 
    point1_norm, T1 = normalize_points(img1_points)
    point2_norm, T2 = normalize_points(img2_points)
    
    #step 1 is to estimate the fundatmental matrix
    f_matrix = get_F_estimate(point1_norm, point2_norm, T1, T2)
    
    #now get the left and right epipoles
    epipole1, epipole2 = get_epipoles(f_matrix)
    
    #using F and epipoles find the camera matricies P and P'
    P, P_prime = get_P(f_matrix, epipole2)
    
    #get world coordinates uaing P 
    world_coords = []
    for x, x_prime in zip(img1_points, img2_points):
        world_coord = get_world_coords(x,x_prime, P, P_prime)
        world_coords.append(world_coord)
    
    #get the parameters for the cost function 
    parameters = []    
    P_prime = P_prime.ravel()

    for i in P_prime:
        parameters.append(i)
    for i in world_coords:
        parameters.append(i[0])
        parameters.append(i[1])
        parameters.append(i[2])
    
    #using LM refine P' and P 
    lm_out = least_squares(cost_func, parameters, method='lm', 
                           args=[img1_points, img2_points])
    lm_out = lm_out.x
    
    #recreate P from LM
    M_refined = lm_out[0:12]
    M_refined = M_refined / M_refined[-1]
    P_prime_refined = np.reshape(np.array(M_refined), (3,4))
    P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    
    
    #calculate the homographies for rectified image 
    H1, H2 = get_rectficication_homography(img2, epipole2, P, 
                                           P_prime_refined, 
                                           img1_points, 
                                           img2_points)
    
    #apply the rectification homographies 
    new_img1, new_img2 = \
        apply_rectification_homographies(img1, 
                                        img2, 
                                        H1, H2, 
                                        '/home/patelb/ece661/hw09/')
    
    #get the edges using canny 
    canny_output1 = apply_canny(new_img1,1)
    canny_output2 = apply_canny(new_img2,2)
    
    #get the corrsponding points between the images using the 
    # canny output 
    matches = get_correspondences(canny_output1, canny_output2,
                                  new_img1, new_img2)
    
    #display all matches
    display_matches(new_img1, new_img2, matches)
    
    #plot the 3d points
    img1_points = [[36,168],[55,226],[218,311],[216,362],[437,114],
                   [417,165],[277,37],[177,103],[174,124],[229,163],
                   [227,186],[353,119],[350,140],[296,68]]
    img2_points = [[21,230],[50,285],[265,348],[265,397],[424,82],
                   [408,142],[236,32],[158,125],[157,146],[233,181],
                   [235,204],[346,107],[346,128],[271,62]]
    plot_3d(matches, img1_points, img2_points, P, P_prime_refined, H1, H2)
    
    return 


if __name__=="__main__":
    main()