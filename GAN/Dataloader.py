import numpy as np
import torch
import os
import cv2
import math
import datetime
import json
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import scipy.io
from torchvision import transforms

trans = transforms.Compose([transforms.ToTensor()])
 
def get_id(i,max_len = 6):
    curr_len = len(str(i))
    return "0"*max(0,(max_len-curr_len))+str(i)
def get_all_key_points(key_points_path,img_path,num_object,top_key_points =128):
    key_points = []
    for obj_index in range(num_object):
        all_key_points = scipy.io.loadmat(key_points_path[obj_index])['fps']
        #print("ALl key points shape: ",all_key_points.shape)
        num_key_points= all_key_points.shape[0]
        all_key_points_4d = np.hstack((all_key_points,np.ones((num_key_points,1))))
        '''
        print("ALl key points 4d shape: ",all_key_points_4d.shape)
        print("Path:",os.path.join(img_path,"0000"+get_id(obj_index+1),"scene_camera.json"))
        '''
        with open(os.path.join(img_path,get_id(obj_index+1),"scene_camera.json"), 'r') as j:
            all_K = json.loads(j.read())

        all_K = {key:np.resize(all_K[key]["cam_K"],(3,3)) for key in all_K.keys()}
        
        with open(os.path.join(img_path,get_id(obj_index+1),"scene_gt.json"), 'r') as j:
            all_RT = json.loads(j.read())
        
        all_RT = {key:np.hstack((np.resize(all_RT[key][0]["cam_R_m2c"],(3,3)),np.resize(all_RT[key][0]["cam_t_m2c"],(1,3)).T*0.1)) for key in all_RT.keys()}

        img_path_obj = os.path.join(img_path,get_id(obj_index+1),"rgb")
        num_img = len([file for file in os.listdir(img_path_obj) if file.endswith(".png")])
        #print("Shape: ",all_K[str(0)].shape," ",all_RT[str(0)].shape,all_key_points_4d.shape)
        all_key_points_2D = np.array([np.matmul(all_K[str(img_index)],np.matmul(all_RT[str(img_index)],all_key_points_4d.T)).T for img_index in range(num_img)])
        
        division =all_key_points_2D[:,:,2]
        division = np.repeat(division[:,:,np.newaxis],2,axis = 2)
        all_key_points_2D = all_key_points_2D[:,:,:2]/division
        camera_optical_points = np.array([0,0,0,1])
        '''
        print("Shape camera: ",np.matmul(all_RT[str(0)],camera_optical_points).shape," ",all_RT[str(0)].shape," ",camera_optical_points.shape)
        print("Shape camera 2: ",all_K[str(0)].shape)
        print("Shape camera 3: ",np.matmul(all_K[str(0)],np.matmul(all_RT[str(0)],camera_optical_points)).shape)
        '''
        all_camera_optical_points = np.array([np.matmul(all_K[str(img_index)],np.matmul(all_RT[str(img_index)],camera_optical_points))  for img_index in range(num_img)])
        division_optical =all_camera_optical_points[:,2]
        division_optical = np.repeat(division_optical[:,np.newaxis],2,axis = 1)
        all_camera_optical_points = all_camera_optical_points[:,:2]/division_optical
        #print("Camera point: ",all_camera_optical_points.shape)
        for img_index in range(num_img):
            curr_optical = all_camera_optical_points[img_index]
            '''
            print("Curr optical: ",curr_optical)
            print("Key point 2d shape: ",all_key_points_2D[img_index].shape,"    ",)
            '''
            distance = all_key_points_2D[img_index] - all_camera_optical_points[img_index]
            #print("Distance: ",distance.shape)
            distance = np.linalg.norm(distance,axis = 1)
            '''
            print("Size distance: ",distance.shape)
            print("Num img: ",num_img)
            '''
            sorted_index = sorted([i for i in range(num_key_points)], key = lambda x:distance[x])
            sorted_index_select = sorted_index[:top_key_points]
            key_points.append(all_key_points_2D[img_index][sorted_index_select,:].astype(int))
    return key_points


class Dataset6D(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, img_path,key_points_path,num_objects = 15,trans = trans,type_data = "real"):
        super(Dataset6D,self).__init__()
        self.files = []
        self.all_key_points_path = [os.path.join(key_points_path,"obj"+get_id(i+1,max_len = 2)+"_fps128.mat") for i in range(num_objects)]

        for i in range(num_objects):
            num_file_obj = len([f for f in os.listdir(os.path.join(img_path,get_id(i+1),"rgb")) if f.endswith(".png")])

            self.files += [os.path.join(img_path,get_id(i+1),"rgb",get_id(img_index)+".png") for img_index in range(num_file_obj)]
        
        if type_data == "fake":
            self.key_points = get_all_key_points(self.all_key_points_path,img_path,num_objects)
        else:
            self.key_points =  [ [] for i in range(len(self.files))]

        #print("Shape of keypoints: ",np.array(self.key_points).shape)
        self.trans = trans


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_name = self.files[idx]
        #print("FIle name: ",file_name)
        image = cv2.imread(file_name) 
        #print("Pre shape: ",np.unique(image))
        image = self.trans(image)
        #print("After shape: ",np.unique(image.numpy()))
        key_points = np.array(self.key_points[idx])
        #print("Shape key points: ",np.shape(key_points))
        key_points = torch.from_numpy(key_points)

        return {"img":image,"key_points":key_points,"name":file_name}

'''
traindata_real = "train"
key_points_path = "lm_fps"
train_real = Dataset6D(traindata_real,key_points_path,num_objects = 15,  trans = trans,type_data = "fake")
train_real_loader = torch.utils.data.DataLoader(train_real, batch_size=1, shuffle = False, num_workers = 0)

print("First file: ",train_real.files[0])
for sample in train_real:
    img = np.array(sample["img"].permute(1,2,0).numpy()*255,dtype = np.uint8)
    print("Shape img: ",img.shape)
    print(np.unique(img))
    key_points = sample["key_points"]
    file_name = sample["name"]
    for point in key_points:
        
        
        img = cv2.circle(img = img.copy(),center = (point[0],point[1]), radius = 3,color = (255, 255, 255))
    cv2.imshow(file_name,img)
    a = cv2.waitKey(0)
    if a == 97:
        break
'''