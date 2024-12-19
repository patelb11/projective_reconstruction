# 🛠️ 3D Projective Reconstruction Using Stereo Images  

This repository contains **Homework 9** from the **Purdue ECE 661 - Computer Vision** course taught by **Professor Avinash Kak**. More information about the course can be found on his website: [Avinash Kak's Computer Vision Page](https://engineering.purdue.edu/kak/computervision/).  

## 📋 Project Overview  
This repository implements **3D Projective Reconstruction** from stereo images. The method reconstructs a scene up to a projective distortion, which may result in a visually distorted reconstruction compared to the real scene. By using constraints derived from the geometry of stereo images, we:  
- Rectify stereo images so that corresponding points lie on the same row.  
- Automatically detect and match interest points using metrics like **Sum of Squared Differences (SSD)** or **Normalized Cross-Correlation (NCC)**.  
- Perform a projective reconstruction of 3D points using rectified correspondences and canonical projection matrices.  
- Visualize the reconstructed 3D points and camera poses.  

## 🛠️ Setup  
1. Place at least two stereo images of the same scene in the same directory as the `hw9.py` file.  
2. Provide at least 8 corresponding points between the two images in the `main()` function for initial estimation.  
3. Run the script to perform image rectification, interest point matching, and 3D projective reconstruction.  

## 🚀 Tech Stack  
- **Programming Language**: Python  
- **Libraries Used**: OpenCV, Matplotlib, NumPy  

## 📷 Images  
Below are examples of input and output images processed during this project:  

### Input Stereo Images  
![Input Image 1](input1.jpg)  
![Input Image 2](input2.jpg)  

### Rectified Images  
![Rectified Image 1](rectified1.jpg)  
![Rectified Image 2](rectified2.jpg)  

### 3D Reconstruction Output  
![3D Reconstruction](reconstruction.jpg)  
