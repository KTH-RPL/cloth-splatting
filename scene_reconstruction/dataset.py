from torch.utils.data import Dataset
from scene_reconstruction.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.graphics_utils import focal2fov

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args
    ):
        self.dataset = dataset
        self.args = args
        self.viewpoint_ids = [data.view_id for data in dataset]
        self.time_ids = [data.time_id for data in dataset]

    def __getitem__(self, index):
        # try:
        #     image, w2c, time = self.dataset[index]
        #     R,T = w2c
        #     FovX = focal2fov(self.dataset.focal[0], image.shape[2])
        #     FovY = focal2fov(self.dataset.focal[0], image.shape[1])
        # except:
        caminfo = self.dataset[index]
        image = caminfo.image
        R = caminfo.R
        T = caminfo.T
        FovX = caminfo.FovX
        FovY = caminfo.FovY
        time = caminfo.time
        flow = caminfo.flow
        
        view_id = self.viewpoint_ids[index]
        time_id = self.time_ids[index]

        return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                          image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,view_id=view_id,time_id=time_id,flow=flow)
    def __len__(self):
        
        return len(self.dataset)


class MDNerfDataset(Dataset):
    def __init__(
        self,
        dataset,
        args
    ):
        self.dataset = dataset
        self.args = args
        self.viewpoint_ids = np.unique([data.view_id for data in dataset])
        self.time_ids = np.unique([data.time_id for data in dataset])
        self.times = np.unique([data.time for data in dataset])
        self.times.sort()

        self.n_viewpoints = len(self.viewpoint_ids)
        self.n_times = len(self.time_ids)
        
        
        # now order data in a grid of view x time 
        self.ordered_data = np.empty((self.n_viewpoints,self.n_times),dtype=object)
        
        for data in dataset:
            # time id is idx in self.time_ids
            # view id is idx in self.viewpoint_ids
            time_id = np.where(self.time_ids == data.time_id)[0][0]
            view_id = np.where(self.viewpoint_ids == data.view_id)[0][0]
            self.ordered_data[view_id,time_id] = data
        

        
    def __getitem__(self, idx):
        # idx is view_id 
        view_id = idx
        
        if self.n_times >= 3:
            mean_time_id = np.random.randint(1,self.n_times-1)
            all_steps = [self.get_one_item(view_id,mean_time_id-1),
                    self.get_one_item(view_id,mean_time_id),
                    self.get_one_item(view_id,mean_time_id+1)]
        else:
            all_steps = [self.get_one_item(view_id,i) for i in range(self.n_times)]
       
        return all_steps

    def get_one_item(self, view_id, time_id):
        # try:
        #     image, w2c, time = self.ordered_data[view_id,time_id]
        #     R,T = w2c
        #     FovX = focal2fov(self.dataset.focal[0], image.shape[2])
        #     FovY = focal2fov(self.dataset.focal[0], image.shape[1])
        # except:
        caminfo = self.ordered_data[view_id, time_id]
        
        if caminfo is None:
            # find a different view_id with the same time_id
            time_caminfos = self.ordered_data[:, time_id]
            # remove None
            time_caminfos = time_caminfos[time_caminfos is not None]
            # randomly choose one
            caminfo = np.random.choice(time_caminfos)

            if caminfo is None:
                raise ValueError("No cam info found, this should not happen. Something is wrong in the provided data.")

        image = caminfo.image
        R = caminfo.R
        T = caminfo.T
        FovX = caminfo.FovX
        FovY = caminfo.FovY
        time = caminfo.time
        flow = caminfo.flow
        mask = caminfo.mask
        time_id = caminfo.time_id
        view_id = caminfo.view_id
        return Camera(colmap_id=view_id,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                          image_name=f"{view_id}",uid=view_id, view_id=view_id, time_id=time_id, data_device=torch.device("cuda"),time=time,flow=flow,mask=mask)
    def __len__(self):
        
        return self.n_viewpoints