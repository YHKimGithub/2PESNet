import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import time
import h5py
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MHD
from loss_func import MHD_loss_function

#Multi Head Detector
def train_MHD(opt): 
    writer = SummaryWriter()
    model = MHD(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["mhd_training_lr"],weight_decay = opt["mhd_weight_decay"])    
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt["mhd_step_gamma"])
    
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset="train"),
                                                batch_size=1, shuffle=True,
                                                num_workers=0, pin_memory=True,drop_last=True)            
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt,subset=opt['inference_subset']),
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
    loss_function = MHD_loss_function
        
    for n_epoch in range(opt['mhd_epoch']):   
        ################train phase####################
        model.train()
        data = []
        labels = []    
        loss_masks = []        
        epoch_action_loss = 0
        epoch_start_loss = 0
        epoch_end_loss = 0
        epoch_cost = 0
        for n_iter,(input_data,label_action,label_start,label_end) in enumerate(train_loader):
            data.append(input_data)
            label=torch.stack([label_action,label_end], dim=-1)
            labels.append(label)
            label_size = label_end.size()
            loss_masks.append(torch.ones(label_size))
            
            if (n_iter+1)%opt['mhd_batch_size'] == 0 or n_iter+1==len(train_loader):
                max_duration = max([data[i].size(1) for i in range(len(data))])
                
                for i in range(len(data)):
                    data_padding = torch.zeros((1, max_duration-data[i].size(1), data[i].size(2)))
                    data[i]=torch.cat([data[i], data_padding],dim=1)
                    label_padding = torch.zeros((1, max_duration-labels[i].size(1))+ labels[i].size()[2:])
                    labels[i]=torch.cat([labels[i], label_padding],dim=1)
                    loss_mask_size=list(loss_masks[i].size())
                    loss_mask_size[1]=max_duration-loss_mask_size[1]
                    loss_masks[i]=torch.cat([loss_masks[i], torch.zeros(loss_mask_size)], dim=1)
                
                data=torch.cat(data, dim=0)
                labels=torch.cat(labels, dim=0)
                loss_masks=torch.cat(loss_masks, dim=0)
                
                for i in range(0, max_duration, opt['max_rnn_input']):
                    st = i
                    ed = min(max_duration, i+opt['max_rnn_input'])
                    scores, ends = model(data[:,st:ed,:].cuda())
                    
                    scores = torch.mul(scores, loss_masks[:,st:ed].cuda())
                    loss = loss_function(labels[:,st:ed,:,0],scores,opt)
                    cost = loss["cost"] 
                    
                    ends = torch.mul(ends, loss_masks[:,st:ed].cuda())
                    loss = loss_function(labels[:,st:ed,:,1],ends,opt)
                    cost_end = loss["cost"] 
                    
                    total_cost=cost+cost_end
                    
                    optimizer.zero_grad()
                    cost.backward()
                    cost_end.backward()
                    optimizer.step()
                    epoch_cost = total_cost.cpu().detach().numpy()
    
                data = []
                labels = []     
                loss_masks = []      
            
        writer.add_scalars('data/cost', {'train': epoch_cost/(n_iter+1)}, n_epoch)
        
        print( "MHD training loss(epoch %d): end - %.03f" %(n_epoch,epoch_cost/(n_iter+1)))
        
        scheduler.step(epoch_cost/(n_iter+1))
        
        ##################validation phase#################
        model.eval()
        data = []
        labels = []    
        loss_masks = []        
        epoch_action_loss = 0
        epoch_start_loss = 0
        epoch_end_loss = 0
        epoch_cost = 0
        for n_iter,(input_data,label_action,label_start,label_end) in enumerate(test_loader):
            data.append(input_data)
            label=torch.stack([label_action,label_end], dim=-1)
            labels.append(label)
            label_size = label_end.size()
            loss_masks.append(torch.ones(label_size))
                        
            if (n_iter+1)%opt['mhd_batch_size'] == 0 or n_iter+1==len(test_loader):
                max_duration = max([data[i].size(1) for i in range(len(data))])
                
                for i in range(len(data)):
                    data_padding = torch.zeros((1, max_duration-data[i].size(1), data[i].size(2)))
                    data[i]=torch.cat([data[i], data_padding],dim=1)
                    label_padding = torch.zeros((1, max_duration-labels[i].size(1))+ labels[i].size()[2:])
                    labels[i]=torch.cat([labels[i], label_padding],dim=1)
                    loss_mask_size=list(loss_masks[i].size())
                    loss_mask_size[1]=max_duration-loss_mask_size[1]
                    loss_masks[i]=torch.cat([loss_masks[i], torch.zeros(loss_mask_size)], dim=1)
                
                data=torch.cat(data, dim=0)
                labels=torch.cat(labels, dim=0)
                loss_masks=torch.cat(loss_masks, dim=0)
                
                for i in range(0, max_duration, opt['max_rnn_input']):
                    st = i
                    ed = min(max_duration, i+opt['max_rnn_input'])
                    scores,ends = model(data[:,st:ed,:].cuda())
                    scores = torch.mul(scores, loss_masks[:,st:ed].cuda())
                    ends = torch.mul(ends, loss_masks[:,st:ed].cuda())
                    
                    loss = loss_function(labels[:,st:ed,:,0],scores,opt)
                    cost = loss["cost"] 
                    
                    loss = loss_function(labels[:,st:ed,:,1],ends,opt)
                    cost_end = loss["cost"] 
                    
                    total_cost=cost_end
                    epoch_cost=total_cost.cpu().detach().numpy()
    
                data = []
                labels = []     
                loss_masks = []      
            
        writer.add_scalars('data/cost', {'test': epoch_cost/(n_iter+1)}, n_epoch)
        print( "MHD testing  loss(epoch %d): end - %.03f" %(n_epoch,epoch_cost/(n_iter+1)))
          
        #################save checkpoint################                                                                              
        state = {'epoch': n_epoch + 1,
                    'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"]+"/mhd_checkpoint.pth.tar" )
        if epoch_cost< model.module.mhd_best_loss:
            model.module.mhd_best_loss = np.mean(epoch_cost)
            torch.save(state, opt["checkpoint_path"]+"/mhd_best.pth.tar" )
        writer.close()
 
def test_MHD(opt): 
    model = MHD(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/mhd_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
                       
    index_list = []
    data = []
    
    outfile = h5py.File('./output/MHD_results.h5', 'w')
    labelfile = h5py.File('./output/MHD_results_label.h5', 'w')
    labels=[]
    
    start_time = time.time()
    total_frames =0 
    for n_iter,(index,input_data,video_feature_frame) in enumerate(test_loader):
        index_list.append(index)
        data.append(input_data)
        label=dataset._get_train_label(index[0],video_feature_frame[0])
        labels.append(torch.stack(label, dim=-1))
        
        if (n_iter+1)%opt['mhd_batch_size'] == 0 or n_iter+1==len(test_loader):
            durations = [data[i].size(1) for i in range(len(data))]
            max_duration = max([data[i].size(1) for i in range(len(data))])
            total_frames += max_duration
            
            for i in range(len(data)):
                data_padding = torch.zeros((1, max_duration-data[i].size(1), data[i].size(2)))
                data[i]=torch.cat([data[i], data_padding],dim=1)
            
            data=torch.cat(data, dim=0)
            
            MHD_output=[]
            MHD_ends=[]
            for i in range(0, max_duration, opt['max_rnn_input']):
                st = i
                ed = min(max_duration, i+opt['max_rnn_input'])
                mini_output, mini_ends = model(data[:,st:ed,:].cuda())
                mini_output = torch.softmax(mini_output, dim=-1).detach().cpu().numpy()
                mini_ends = torch.softmax(mini_ends, dim=-1).detach().cpu().numpy()
                MHD_output.append(mini_output)
                MHD_ends.append(mini_ends)
            
            MHD_output=np.concatenate(MHD_output, axis=1)
            MHD_ends=np.concatenate(MHD_ends, axis=1)
            
            for batch_idx, full_idx in enumerate(index_list):
                video = test_loader.dataset.video_list[full_idx]
                video_result = MHD_output[batch_idx, :durations[batch_idx],:]
                video_result_end = MHD_ends[batch_idx, :durations[batch_idx],:]
                    
                dset_results = outfile.create_dataset(video+'/score', video_result.shape, maxshape=video_result.shape, chunks=True, dtype=np.float32)
                dset_results[:,:] = video_result[:,:]   
                dset_ends = outfile.create_dataset(video+'/end', video_result.shape, maxshape=video_result.shape, chunks=True, dtype=np.float32)
                dset_ends[:,:] = video_result_end[:,:]   
                label=labels[batch_idx].numpy()
                dset_labels = labelfile.create_dataset(video, label.shape, maxshape=label.shape, chunks=True, dtype=np.float32)
                dset_labels[:,:] = label[:,:]                        

            index_list =[]
            data = []
            labels = []
            
    end_time = time.time()
    working_time = end_time-start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    

def generate_MHD_dataset(opt): 
    model = MHD(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/mhd_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    #make train result file
    start_time = time.time()
    dataset = VideoDataSet(opt,subset='train')
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
                    
    index_list = []
    data = []
    
    outfile = h5py.File('./output/MHD_results_train.h5', 'w')
    labelfile = h5py.File('./output/MHD_results_label_train.h5', 'w')
    labels=[]
    
    total_frames =0 
    for n_iter,(index,input_data,video_feature_frame) in enumerate(test_loader):
        index_list.append(index)
        data.append(input_data)
        label=dataset._get_train_label(index[0],video_feature_frame[0])
        labels.append(torch.stack(label, dim=-1))
        
        if (n_iter+1)%opt['mhd_batch_size'] == 0 or n_iter+1==len(test_loader):
            durations = [data[i].size(1) for i in range(len(data))]
            max_duration = max([data[i].size(1) for i in range(len(data))])
            total_frames += max_duration
            
            for i in range(len(data)):
                data_padding = torch.zeros((1, max_duration-data[i].size(1), data[i].size(2)))
                data[i]=torch.cat([data[i], data_padding],dim=1)
            
            data=torch.cat(data, dim=0)
            
            MHD_output=[]
            MHD_ends=[]
            for i in range(0, max_duration, opt['max_rnn_input']):
                st = i
                ed = min(max_duration, i+opt['max_rnn_input'])
                mini_output, mini_ends = model(data[:,st:ed,:].cuda())
                mini_output = torch.softmax(mini_output, dim=-1).detach().cpu().numpy()
                mini_ends = torch.softmax(mini_ends, dim=-1).detach().cpu().numpy()
                MHD_output.append(mini_output)
                MHD_ends.append(mini_ends)
            
            MHD_output=np.concatenate(MHD_output, axis=1)
            MHD_ends=np.concatenate(MHD_ends, axis=1)
            
            for batch_idx, full_idx in enumerate(index_list):
                video = test_loader.dataset.video_list[full_idx]
                video_result = MHD_output[batch_idx, :durations[batch_idx],:]
                video_result_end = MHD_ends[batch_idx, :durations[batch_idx],:]
                    
                dset_results = outfile.create_dataset(video+'/score', video_result.shape, maxshape=video_result.shape, chunks=True, dtype=np.float32)
                dset_results[:,:] = video_result[:,:]   
                dset_ends = outfile.create_dataset(video+'/end', video_result.shape, maxshape=video_result.shape, chunks=True, dtype=np.float32)
                dset_ends[:,:] = video_result_end[:,:]   
                label=labels[batch_idx].numpy()
                dset_labels = labelfile.create_dataset(video, label.shape, maxshape=label.shape, chunks=True, dtype=np.float32)
                dset_labels[:,:] = label[:,:]                        

            index_list =[]
            data = []
            labels = []
            
    end_time = time.time()
    working_time = end_time-start_time
    print("trainset working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    
    #make test result file
    start_time = time.time()
    dataset = VideoDataSet(opt,subset='test')
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
                    
    index_list = []
    data = []
    
    outfile = h5py.File('./output/MHD_results_test.h5', 'w')
    labelfile = h5py.File('./output/MHD_results_label_test.h5', 'w')
    labels=[]
    
    total_frames =0 
    for n_iter,(index,input_data,video_feature_frame) in enumerate(test_loader):
        index_list.append(index)
        data.append(input_data)
        label=dataset._get_train_label(index[0],video_feature_frame[0])
        labels.append(torch.stack(label, dim=-1))
        
        if (n_iter+1)%opt['mhd_batch_size'] == 0 or n_iter+1==len(test_loader):
            durations = [data[i].size(1) for i in range(len(data))]
            max_duration = max([data[i].size(1) for i in range(len(data))])
            total_frames += max_duration
            
            for i in range(len(data)):
                data_padding = torch.zeros((1, max_duration-data[i].size(1), data[i].size(2)))
                data[i]=torch.cat([data[i], data_padding],dim=1)
            
            data=torch.cat(data, dim=0)
            
            MHD_output=[]
            MHD_ends=[]
            for i in range(0, max_duration, opt['max_rnn_input']):
                st = i
                ed = min(max_duration, i+opt['max_rnn_input'])
                mini_output, mini_ends = model(data[:,st:ed,:].cuda())
                mini_output = torch.softmax(mini_output, dim=-1).detach().cpu().numpy()
                mini_ends = torch.softmax(mini_ends, dim=-1).detach().cpu().numpy()
                MHD_output.append(mini_output)
                MHD_ends.append(mini_ends)
            
            MHD_output=np.concatenate(MHD_output, axis=1)
            MHD_ends=np.concatenate(MHD_ends, axis=1)
            
            for batch_idx, full_idx in enumerate(index_list):
                video = test_loader.dataset.video_list[full_idx]
                video_result = MHD_output[batch_idx, :durations[batch_idx],:]
                video_result_end = MHD_ends[batch_idx, :durations[batch_idx],:]
                    
                dset_results = outfile.create_dataset(video+'/score', video_result.shape, maxshape=video_result.shape, chunks=True, dtype=np.float32)
                dset_results[:,:] = video_result[:,:]   
                dset_ends = outfile.create_dataset(video+'/end', video_result.shape, maxshape=video_result.shape, chunks=True, dtype=np.float32)
                dset_ends[:,:] = video_result_end[:,:]   
                label=labels[batch_idx].numpy()
                dset_labels = labelfile.create_dataset(video, label.shape, maxshape=label.shape, chunks=True, dtype=np.float32)
                dset_labels[:,:] = label[:,:]                        

            index_list =[]
            data = []
            labels = []
            
    end_time = time.time()
    working_time = end_time-start_time
    print("testset working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))

