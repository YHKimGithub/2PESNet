import numpy as np
import random
import h5py
import json
import torch
import torch.utils.data as data
import os

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
 
#for Multi head detector data         
class VideoDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.temporal_interval = opt["temporal_interval"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_rgb_train_path = opt["video_feature_rgb_train"]
        self.feature_flow_train_path = opt["video_feature_flow_train"]
        self.feature_rgb_test_path = opt["video_feature_rgb_test"]
        self.feature_flow_test_path = opt["video_feature_flow_test"]
        self.video_anno_path = opt["video_anno"]        
        self.num_of_class = opt["num_of_class"]
        self.label_name = [None for i in range(0,self.num_of_class)]        
        
        if self.subset == "train":
            self.feature_rgb_file = h5py.File(self.feature_rgb_train_path, 'r')
            self.feature_flow_file = h5py.File(self.feature_flow_train_path, 'r')
        else:
            self.feature_rgb_file = h5py.File(self.feature_rgb_test_path, 'r')
            self.feature_flow_file = h5py.File(self.feature_flow_test_path, 'r')    
        self._getDatasetDict()
    
    #read json annot file & construct data structure    
    def _getDatasetDict(self):
        anno_database= load_json(self.video_anno_path)
        anno_database=anno_database['database']
        self.video_dict = {}
        for video_name in anno_database:
            video_info=anno_database[video_name]
            video_subset=anno_database[video_name]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
            
            for seg in video_info['annotations']:
                self.label_name[seg['labelIndex']] = seg['label']
                
        self.video_list = list(self.video_dict.keys())
        print ("%s subset video numbers: %d" %(self.subset,len(self.video_list)))

    #read an item(#index) from the data structure 
    def __getitem__(self, index):
        video_data, video_feature_frame = self._get_base_data(index,self.mode)
        if self.mode == "train":
            match_score_action,match_score_start,match_score_end =  self._get_train_label(index,video_feature_frame)
            jitter=0
            if(self.subset=="train"):
                jitter = random.randrange(0, video_feature_frame*8//10)
            return video_data[jitter:],match_score_action[jitter:],match_score_start[jitter:],match_score_end[jitter:]
        else:
            return index,video_data,video_feature_frame
            
    #read feature from the feature file 
    def _get_base_data(self,index, mode):  
        video_name=self.video_list[index]        
        
        feature_rgb = self.feature_rgb_file[video_name]
        feature_flow = self.feature_flow_file[video_name]
        duration = min(len(feature_rgb), len(feature_flow))
        if(duration == 0):
            print(video_name)
        feature = np.append(feature_rgb[:duration,:],feature_flow[:duration,:], axis=1)
        feature = torch.from_numpy(np.array(feature))
        
        return feature, duration
     
    #make start&end&action label array from the annotation data structure
    def _get_train_label(self,index,video_feature_frame): 
        video_name=self.video_list[index]
        video_info=self.video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration']
        video_labels=video_info['annotations']
        
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment_frame'][0]
            tmp_end=min(tmp_info['segment_frame'][1],video_feature_frame*self.temporal_interval)
            tmp_label=tmp_info['labelIndex']
            gt_bbox.append([tmp_start,tmp_end,tmp_label])
            
        gt_bbox=np.array(gt_bbox)
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]

        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(2*self.temporal_interval,np.round(self.boundary_ratio*gt_lens)).astype('int')
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small//2,gt_xmins+gt_len_small//2),axis=1)
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small//2,gt_xmaxs+gt_len_small//2),axis=1)
        
        match_score_action=np.zeros((video_feature_frame,self.num_of_class), dtype=np.float32)
        match_score_action[:,-1]=1
        for idx in range(gt_bbox.shape[0]):
            match_score_action[gt_bbox[idx,0]//self.temporal_interval:gt_bbox[idx,1]//self.temporal_interval, gt_bbox[idx,2]]=1
            match_score_action[gt_bbox[idx,0]//self.temporal_interval:gt_bbox[idx,1]//self.temporal_interval, -1]=0
        match_score_start=np.zeros((video_feature_frame,self.num_of_class), dtype=np.float32)
        match_score_start[:,-1]=1
        for idx in range(gt_start_bboxs.shape[0]):
            match_score_start[gt_start_bboxs[idx,0]//self.temporal_interval:gt_start_bboxs[idx,1]//self.temporal_interval+1, gt_bbox[idx,2]]=1
            match_score_start[gt_start_bboxs[idx,0]//self.temporal_interval:gt_start_bboxs[idx,1]//self.temporal_interval+1, -1]=0
        match_score_end=np.zeros((video_feature_frame,self.num_of_class), dtype=np.float32)
        match_score_end[:,-1]=1
        for idx in range(gt_end_bboxs.shape[0]):
            match_score_end[gt_end_bboxs[idx,0]//self.temporal_interval:gt_end_bboxs[idx,1]//self.temporal_interval+1, gt_bbox[idx,2]]=1
            match_score_end[gt_end_bboxs[idx,0]//self.temporal_interval:gt_end_bboxs[idx,1]//self.temporal_interval+1, -1]=0
        
        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action,match_score_start,match_score_end
    
    def __len__(self):
        return len(self.video_list)
 
#for Action Start Detection data       
class VideoDataSet_Inverse(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.temporal_interval = opt["temporal_interval"]
        self.boundary_ratio = opt["boundary_ratio"]
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_rgb_train_path = opt["video_feature_rgb_train"]
        self.feature_flow_train_path = opt["video_feature_flow_train"]
        self.feature_rgb_test_path = opt["video_feature_rgb_test"]
        self.feature_flow_test_path = opt["video_feature_flow_test"]
        self.video_anno_path = opt["video_anno"]        
        self.num_of_class = opt["num_of_class"]        
        
        if self.subset == "train":
            self.feature_rgb_file = h5py.File(self.feature_rgb_train_path, 'r')
            self.feature_flow_file = h5py.File(self.feature_flow_train_path, 'r')
        else:
            self.feature_rgb_file = h5py.File(self.feature_rgb_test_path, 'r')
            self.feature_flow_file = h5py.File(self.feature_flow_test_path, 'r')    
        self._getDatasetDict()
    
    #read json annot file & construct data structure     
    def _getDatasetDict(self):
        anno_database= load_json(self.video_anno_path)
        anno_database=anno_database['database']
        self.video_dict = {}
        for video_name in anno_database:
            video_info=anno_database[video_name]
            video_subset=anno_database[video_name]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        self.seg_list = []
        for video_name in self.video_list:
            for seg_info in anno_database[video_name]['annotations']:
                if( seg_info['segment_frame'][1] > anno_database[video_name]['duration_frame']):
                    continue
                self.seg_list.append([video_name, seg_info['segment_frame'][0], seg_info['segment_frame'][1], seg_info['labelIndex']])
        print ("%s subset video seg numbers: %d" %(self.subset,len(self.seg_list)))

    #read an inversed feature & label array from the data structure
    def __getitem__(self, index):
        seg_info = self.seg_list[index]  
        video_name = seg_info[0]        
        seg_st = seg_info[1] // self.temporal_interval
        seg_ed = seg_info[2] // self.temporal_interval
        seg_len= seg_ed-seg_st
        label_idx = seg_info[3]
        
        feature_rgb = self.feature_rgb_file[video_name]
        feature_flow = self.feature_flow_file[video_name]
        duration = min(len(feature_rgb), len(feature_flow))
        
        jitter=0
        if(self.subset=="train"):
            jitter = random.randrange(0, max(1,seg_len//2))
                
        st = max(0, seg_st-jitter)
        ed = min(duration, seg_ed+jitter)
        
        feature = np.append(feature_rgb[st:ed,:],feature_flow[st:ed,:], axis=1)
        feature = torch.from_numpy(np.array(feature[::-1,:]))
                
        label_action = np.zeros((ed-st,self.num_of_class), dtype=np.float32)
        label_action[:,-1]=1
        label_action[jitter:ed-st-jitter,label_idx]=1
        label_action[jitter:ed-st-jitter,-1]=0
        
        label_start = np.zeros((ed-st,self.num_of_class), dtype=np.float32)
        label_start[:,-1]=1
        gt_bdry_len=np.maximum(1, np.round(self.boundary_ratio*seg_len)).astype('int')
        gt_start = np.maximum(jitter-gt_bdry_len,0).astype('int')
        gt_end = jitter+gt_bdry_len
        label_start[gt_start:gt_end,label_idx]=1
        label_start[gt_start:gt_end,-1]=0
        
        label_action= torch.from_numpy(np.array(label_action[::-1,:]))
        label_start= torch.from_numpy(np.array(label_start[::-1,:]))
        
        return feature, label_action, label_start
    
    def __len__(self):
        return len(self.seg_list)
 
#for End detection refinement data        
class DistribDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.temporal_interval = opt["temporal_interval"]
        self.subset = subset
        self.mode = opt["mode"]
        self.video_anno_path = opt["video_anno"]        
        self.num_of_class = opt["num_of_class"]        
        
        if self.subset == "train":
            self.end_detection_file = h5py.File('./output/MHD_results_train.h5', 'r')
        else:
            self.end_detection_file = h5py.File('./output/MHD_results_test.h5', 'r')
        self._getDatasetDict()
    
    #read json annot file & construct data structure    
    def _getDatasetDict(self):
        anno_database= load_json(self.video_anno_path)
        anno_database=anno_database['database']
        self.video_dict = {}
        for video_name in anno_database:
            video_info=anno_database[video_name]
            video_subset=anno_database[video_name]['subset']
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
                
        self.video_list = list(self.video_dict.keys())
        print ("%s subset video numbers: %d" %(self.subset,len(self.video_list)))

    #read an item(#index) from the data structure
    def __getitem__(self, index):
        video_data, video_feature_frame = self._get_base_data(index,self.mode, False)
        if self.mode == "train":
            match_score_end = self._get_train_label(index,video_feature_frame)
            return video_data,match_score_end
        else:
            return index,video_data,video_feature_frame
    
    #make the tensor of multi-head scores from the MHD result files        
    def _get_base_data(self,index, mode, normalization=False):  
        video_name=self.video_list[index]        
        
        distrib_end = self.end_detection_file[video_name+'/end']
        distrib_score = self.end_detection_file[video_name+'/score']
        duration = len(distrib_end)
        if(duration == 0):
            print(video_name)
        if(normalization):
            np_end=np.array(distrib_end[:,:-1])
            np_score=np.array(distrib_score[:,:-1])
            distrib_end = (np_end - np_end.min()) / (np_end.max() - np_end.min())  
            distrib_score = (np_score - np_score.min()) / (np_score.max() - np_score.min())  
            distrib = np.stack((distrib_end,distrib_score), axis=-1) 
        else:
            distrib = np.stack((distrib_end[:,:-1],distrib_score[:,:-1]), axis=-1) 
        distrib = torch.from_numpy(np.array(distrib))
        
        return distrib, duration
    
    #make action end label array from the annotation data structure
    def _get_train_label(self,index,video_feature_frame): 
        video_name=self.video_list[index]
        video_info=self.video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration']
        video_labels=video_info['annotations']
        
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment_frame'][0]
            tmp_end=min(tmp_info['segment_frame'][1],video_feature_frame*self.temporal_interval)
            tmp_label=tmp_info['labelIndex']
            gt_bbox.append([tmp_start,tmp_end,tmp_label])
            
        gt_bbox=np.array(gt_bbox)
        match_score_end=np.zeros((video_feature_frame,self.num_of_class), dtype=np.float32)
        match_score_end[:,-1]=1
        for idx in range(gt_bbox.shape[0]):
            match_score_end[gt_bbox[idx,1]//self.temporal_interval-1:gt_bbox[idx,1]//self.temporal_interval+2, gt_bbox[idx,2]]=1
            match_score_end[gt_bbox[idx,1]//self.temporal_interval-1:gt_bbox[idx,1]//self.temporal_interval+2, -1]=0
        
        match_score_end = torch.Tensor(match_score_end)
        return match_score_end
    
    def __len__(self):
        return len(self.video_list)
