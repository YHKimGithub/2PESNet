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
from dataset import DistribDataSet
from models import EDR
from loss_func import EDR_loss_function

#End Detection Refinement
def train_EDR(opt): 
    writer = SummaryWriter()
    model = EDR(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["edr_training_lr"],weight_decay = opt["edr_weight_decay"])    
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt["edr_step_gamma"])
    
    train_loader = torch.utils.data.DataLoader(DistribDataSet(opt,subset="train"),
                                                batch_size=1, shuffle=True,
                                                num_workers=0, pin_memory=True,drop_last=True)            
    
    test_loader = torch.utils.data.DataLoader(DistribDataSet(opt,subset=opt['inference_subset']),
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
                                              
    loss_function = EDR_loss_function
    
    for n_epoch in range(opt['edr_epoch']):   
        ################train phase####################
        model.train()     
        epoch_cost = 0
        for n_iter,(input_data,label) in enumerate(train_loader):
            input_data=input_data[0]
            label=label[0].cuda()
            duration=input_data.size(0)  
            saved_probs = []
            rewards = []
                        
            for i in range(0, duration):
                sub_input=[]
                for j in range(1-opt['edr_input_score_size'],1):
                    if( i+j <0):
                        sub_input.append(input_data[0])
                    else:
                        sub_input.append(input_data[i+j])
                        
                sub_input=torch.cat(sub_input,dim=-1)       
                sub_input=sub_input.reshape(1,-1)         
                activ_prob = model(sub_input.cuda())
                
                saved_probs.append(activ_prob)
                
            saved_probs = torch.stack(saved_probs,dim=0)
            saved_probs = saved_probs.reshape(-1,saved_probs.size()[-1])
            loss = loss_function(label,saved_probs,opt)
            loss = loss["cost"]
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_cost += loss.item()
            
        writer.add_scalars('data/cost', {'train': epoch_cost}, n_epoch)
        
        print( "EDR training loss(epoch %d): end - %.03f" %(n_epoch,epoch_cost))
        
        scheduler.step(epoch_cost/(n_iter+1))
        
        ##################validation phase#################
        model.eval()
        epoch_cost = 0
        for n_iter,(input_data,label) in enumerate(test_loader):
            input_data=input_data[0]
            label=label[0].cuda()
            duration=input_data.size(0)  
            saved_probs = []
            rewards = []
            
            for i in range(0, duration):
                sub_input=[]
                for j in range(1-opt['edr_input_score_size'],1):
                    if( i+j <0):
                        sub_input.append(input_data[0])
                    else:
                        sub_input.append(input_data[i+j])
                        
                sub_input=torch.cat(sub_input,dim=-1)
                sub_input=sub_input.reshape(1,-1)                         
                activ_prob = model(sub_input.cuda())
                
                saved_probs.append(activ_prob)
                
            saved_probs = torch.stack(saved_probs,dim=0)
            saved_probs = saved_probs.reshape(-1,saved_probs.size()[-1])
            loss = loss_function(label,saved_probs,opt)
            loss = loss['cost']
                            
            epoch_cost += loss.item()
            
        writer.add_scalars('data/cost', {'test': epoch_cost}, n_epoch)
        
        print( "EDR testing  loss(epoch %d): end - %.03f" %(n_epoch,epoch_cost))
         
        #################save checkpoint################                                                                               
        state = {'epoch': n_epoch + 1,
                    'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"]+"/edr_checkpoint.pth.tar" )
        if epoch_cost< model.module.edr_best_loss:
            model.module.edr_best_loss = np.mean(epoch_cost)
            torch.save(state, opt["checkpoint_path"]+"/edr_best.pth.tar" )
        writer.close()

def test_EDR(opt): 
    model = EDR(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/edr_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()
    
    dataset = DistribDataSet(opt,subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
                   
    outfile = h5py.File('./output/EDR_results.h5', 'w')
    
    start_time = time.time()
    total_frames =0 
    for n_iter,(index,input_data,video_feature_frame) in enumerate(test_loader):
                        
        input_data=input_data[0]
        label=dataset._get_train_label(index[0],video_feature_frame[0])
        duration=input_data.size(0)  
        
        video_name = test_loader.dataset.video_list[index]
        dset_results = outfile.create_dataset(video_name, label.shape[:2], maxshape=label.shape[:2], chunks=True, dtype=np.float32)
        
        for i in range(0, duration):
            sub_input=[]
            for j in range(-4,1):
                if( i+j <0):
                    sub_input.append(input_data[0])
                else:
                    sub_input.append(input_data[i+j])
                    
            sub_input=torch.cat(sub_input,dim=-1)
            sub_input=sub_input.reshape(1,-1)         
            activ_prob = model(sub_input.cuda())
            activ_prob = torch.softmax(activ_prob,dim=-1)
            activ_prob = activ_prob.detach().cpu().numpy()
            dset_results[i,:] = activ_prob[0,:]   
             
    end_time = time.time()
    working_time = end_time-start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))


