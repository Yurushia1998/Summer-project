
import torch
from torch import nn
from torchvision import transforms
import torch.optim as optim
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from models import Generator, Discriminator, KeyPointDetector 
from Dataloader import Dataset6D
import warnings
import math
import numpy as np
import os
import json
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
batches = 8
num_epoches = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
is_cuda = torch.cuda.is_available()
model_save_path = "Models/Final_model_"+str(num_epoches)+".pth"
trans = transforms.Compose([transforms.ToTensor()])



train_data_real_path =  "/Users/macbookpro/Downloads/Adelaide_Academic_Year_2020/Summer project/Bo Chen/Summer-project/GAN_data/real"
test_data_real_path =  "/Users/macbookpro/Downloads/Adelaide_Academic_Year_2020/Summer project/Bo Chen/Summer-project/GAN_data/test_real"
test_data_fake_path =  "/Users/macbookpro/Downloads/Adelaide_Academic_Year_2020/Summer project/Bo Chen/Summer-project/GAN_data/test_fake"

key_points_path = "lm_fps"
train_data_fake_path = "/Users/macbookpro/Downloads/Adelaide_Academic_Year_2020/Summer project/Bo Chen/Summer-project/GAN_data/fake"
output_save_path = "Output_test"
load_path = "Models/Final_model_"+str(num_epoches)+".pth"

train_real =  Dataset6D(train_data_real_path,key_points_path,num_objects = 15,   trans = trans,type_data = "real")
train_fake= Dataset6D(train_data_fake_path,key_points_path, num_objects = 15,  trans = trans,type_data = "fake")
test_real = Dataset6D(test_data_real_path,key_points_path,num_objects = 15,   trans = trans,type_data = "real")
test_fake = Dataset6D(test_data_fake_path,key_points_path,num_objects = 15,   trans = trans,type_data = "real")


train_real_loader = torch.utils.data.DataLoader(train_real, batch_size=batches, shuffle = False, num_workers = 0)
train_fake_loader =torch.utils.data.DataLoader(train_fake, batch_size=batches, shuffle = False, num_workers = 0)
test_real_loader = torch.utils.data.DataLoader(test_real, batch_size=1, shuffle = False, num_workers = 0)
test_fake_loader = torch.utils.data.DataLoader(test_fake, batch_size=1, shuffle = False, num_workers = 0)



discriminator = Discriminator()
generator = Generator()
keypoint_detector = KeyPointDetector()


if is_cuda:
    discriminator.to(device)
    generator.to(device)
    keypoint_detector.to(device)
    print("Running with cuda")
else:
    print("Running with cpu")

criterion = nn.BCELoss()

criterion_keypoints = nn.MSELoss()
discrim_optim = optim.Adam(discriminator.parameters(), lr= 0.0002)
generat_optim = optim.Adam(generator.parameters(), lr=0.0002)
keypoint_detector_optim = optim.Adam(keypoint_detector.parameters(), lr=0.0002)

def noise(x,y):
    if is_cuda:
        return torch.randn(x,y).cuda()
    return torch.randn(x,y)

def get_nearones(x):
    if is_cuda:
        return torch.ones(x,1).cuda()-0.01
    return torch.ones(x,1)-0.01

def get_nearzeros(x):
    if is_cuda:
        return torch.zeros(x,1).cuda()+0.01
    return torch.zeros(x,1)+0.01

def plotimage(is_cuda):
    if is_cuda:
        plt.imshow(generator(noise(1, 128)).cpu().detach().view(28,28).numpy(), cmap=cm.gray)
    else:
        plt.imshow(generator(noise(1, 128)).detach().view(28,28).numpy(), cmap=cm.gray)
    plt.show()

derrors = []
gerrors = []
keyerrors = []
dxcumul = []
gxcumul = []

def train():
    for epoch in range(num_epoches):
        dx = 0
        gx = 0
        derr = 0
        gerr = 0
        keyerr = 0
        num_iterations = math.floor(train_real.__len__()/batches)
        #print("Number_iteration: ",num_iterations," ",train_real.__len__()," ",train_fake.__len__())
       
        dataloader_real_iterator = iter(train_real_loader)
        dataloader_fake_iterator = iter(train_fake_loader)

        for i in range(num_iterations):
            discriminator.train()
            generator.train()
            keypoint_detector.train()
            pos_sample_real = next(dataloader_real_iterator)["img"]
            pos_sample_fake = next(dataloader_fake_iterator)
            label_keypoints = pos_sample_fake["key_points"]
            #print("Before: ",label_keypoints.shape)
            
            label_keypoints_flatten = label_keypoints.view(-1,256).float()
            #print("After: ",label_keypoints_flatten.shape)
            pos_sample_fake = pos_sample_fake["img"]
            # Training Discriminator network
            discrim_optim.zero_grad()
            generator.require_gradient = False
            pos_sample_real = pos_sample_real.to(device) 
            pos_sample_fake = pos_sample_fake.to(device) 
            label_keypoints_flatten = label_keypoints_flatten.to(device) 
            '''
            print("SHape real: ",pos_sample_real.size())
            print("SHape keypoints: ",label_keypoints.size())
            print("SHape fake: ",pos_sample_fake.size())
            '''
            pos_sample = generator(pos_sample_real)
            #print("Shape generator: ",pos_sample.size())
            pos_predicted = discriminator(pos_sample)
            pos_error = criterion(pos_predicted, get_nearones(batches))

            '''
            print()
            print("Calling generator")
            '''
            neg_samples = generator(pos_sample_fake)
            #print("GENERATOR: ",neg_samples.size())
            neg_predicted = discriminator(neg_samples)
            neg_error = criterion(neg_predicted, get_nearzeros(batches))
            
            discriminator_error = pos_error + neg_error
            discriminator_error.backward()
            discrim_optim.step()
            
            generator.require_gradient = True
            discriminator.require_gradient = False

            # Training generator network
            generat_optim.zero_grad()
            gen_samples = generator(pos_sample_fake)
            gen_predicted = discriminator(gen_samples)
            generator_error = criterion(gen_predicted, get_nearones(batches))
            generator_error.backward()
            generat_optim.step()
            gen_samples = gen_samples.detach()

            keypoint_detector_optim.zero_grad()
            
            keypoint_predicted = keypoint_detector(gen_samples)
            #print("SHape: ",keypoint_predicted.size()," ",label_keypoints_flatten.size())
            keypoint_detector_error = criterion_keypoints(keypoint_predicted, label_keypoints_flatten)
            keypoint_detector_error.backward()
            keypoint_detector_optim.step()

            discriminator.require_gradient = True
            
            derr += discriminator_error
            gerr += generator_error
            keyerr += keypoint_detector_error
            dx += pos_predicted.data.mean()
            gx += neg_predicted.data.mean()
            #print("End an iteration")
            
        print(f'Epoch:{epoch}.. D x : {dx/10:.4f}.. G x: {gx/10:.4f}.. D err : {derr/10:.4f}.. G err: {gerr/10:.4f}.. K err: {keyerr/10:.4f}')
        
        
        derrors.append(dx/10)
        gerrors.append(gx/10)
        keyerrors.append(keyerr)
        save_state = {"num_epochs":batches,"generator":generator.state_dict(),"discriminator":discriminator.state_dict(),"keypoint_detector":keypoint_detector.state_dict(),"optimizer_generator":generat_optim.state_dict(),
                    "optimizer_discriminator":discrim_optim.state_dict(),"optimizer_keypoint":keypoint_detector_optim.state_dict(),"derrors":derrors,"gerrors":gerrors,"keyerrors":keyerrors}
        torch.save(save_state, model_save_path)


def test(discriminator,generator,keypoint_detector,load_path,test_real_loader,output_save_path,batches = 1):
    num_iterations = math.ceil(test_real_loader.__len__()/batches)
    file = torch.load(load_path,map_location=torch.device('cpu'))
    discriminator.load_state_dict(file["discriminator"])
    generator.load_state_dict(file["generator"])
    keypoint_detector.load_state_dict(file["keypoint_detector"])
    discriminator.eval()
    generator.eval()
    keypoint_detector.eval()
    dataloader_real_iterator = iter(test_real_loader)
    all_result = {}
    curr_obj = 1
    for i in range(num_iterations):
        
        pos_sample_real = next(dataloader_real_iterator)
        
        name = pos_sample_real["name"][0].split("/")[-1].split(".")[0]
        pos_sample_real = pos_sample_real["img"]
        pos_sample_real = pos_sample_real.to(device) 
        pos_sample = generator(pos_sample_real)
        keypoint_predicted = keypoint_detector(pos_sample)
        all_result[name]=keypoint_predicted.cpu().detach().numpy().tolist()
        if i%5 == 4:
            with open(os.path.join(output_save_path,'output_'+str(curr_obj)+'.json'), 'w') as outfile:
                json.dump(all_result, outfile)
            curr_obj += 1
            all_result = {}

def test_embedding(discriminator,generator,keypoint_detector,load_path,test_fake_real_loader,test_real_real_loader,output_save_path,batches = 1):
    num_iterations = math.ceil(test_real_loader.__len__()/batches)
    file = torch.load(load_path,map_location=torch.device('cpu'))
    discriminator.load_state_dict(file["discriminator"])
    generator.load_state_dict(file["generator"])
    keypoint_detector.load_state_dict(file["keypoint_detector"])
    discriminator.eval()
    generator.eval()
    keypoint_detector.eval()
    dataloader_real_iterator = iter(test_real_loader)
    all_result = {}
    curr_obj = -1
    all_real = [[] for i in range(5)]
    all_fake = [[] for i in range(5)]
    #internal_dis_fake = 
    #internal_dis_real = 
    for i in range(num_iterations):
        
        pos_sample_real = next(dataloader_real_iterator)
        
        name = pos_sample_real["name"][0].split("/")[-1].split(".")[0]
        pos_sample_real = pos_sample_real["img"]
        pos_sample_real = pos_sample_real.to(device) 
        pos_sample_real_emb = generator(pos_sample_real)
        pos_sample_real_dis_output = discriminator(pos_sample_real_emb).cpu().detach().numpy()
        pos_sample_real_emb = pos_sample_real_emb.cpu().detach().numpy()


        pos_sample_fake = next(dataloader_real_iterator)
        
        name = pos_sample_fake["name"][0].split("/")[-1].split(".")[0]
        pos_sample_fake = pos_sample_fake["img"]
        pos_sample_fake = pos_sample_fake.to(device) 
        pos_sample_fake_emb = generator(pos_sample_fake)
        pos_sample_fake_dis_output = discriminator(pos_sample_fake_emb).cpu().detach().numpy()
        pos_sample_fake_emb = pos_sample_fake_emb.cpu().detach().numpy()
        print("Value fake: ",pos_sample_fake_emb)
        print("After dis: ",pos_sample_real_dis_output," ",pos_sample_fake_dis_output)
        print("Index: ",i,"Distance is: ",np.linalg.norm(pos_sample_real_emb-pos_sample_fake_emb)," ",np.linalg.norm(pos_sample_real_dis_output-pos_sample_fake_dis_output))
        if i%5 == 0:
            curr_obj += 1
        print("Curr obj: ",curr_obj)
        all_fake[curr_obj].append(pos_sample_fake_emb)
        all_real[curr_obj].append(pos_sample_real_emb)
            
            
#test(discriminator,generator,keypoint_detector,load_path,test_real_loader,output_save_path)
train()
#test_embedding(discriminator,generator,keypoint_detector,load_path,test_fake_loader,test_real_loader,output_save_path)

