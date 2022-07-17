import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import time
import h5py
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet, VideoDataSet_Inverse
from models import MHD, ASD, EDR
from loss_func import ASD_loss_function

#Action Start Detection
def train_ASD(opt): 
    writer = SummaryWriter()
    model = ASD(opt).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=opt["asd_training_lr"],weight_decay = opt["asd_weight_decay"])    
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt["asd_step_gamma"])
    
    train_loader = torch.utils.data.DataLoader(VideoDataSet_Inverse(opt,subset="train"),
                                                batch_size=1, shuffle=True,
                                                num_workers=0, pin_memory=True,drop_last=True)            
    
    test_loader = torch.utils.data.DataLoader(VideoDataSet_Inverse(opt,subset=opt['inference_subset']),
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
    
    loss_function = ASD_loss_function
    
    for n_epoch in range(opt['asd_epoch']):   
        ################train phase####################
        model.train()
        data = []
        labels = []    
        loss_masks = []        
        epoch_action_loss = 0
        epoch_start_loss = 0
        epoch_end_loss = 0
        epoch_cost = 0
        for n_iter,(input_data,label_action,label_start) in enumerate(train_loader):
            data.append(input_data)
            labels.append(label_start)
            label_size = label_start.size()
            loss_masks.append(torch.ones(label_size))
            
            if (n_iter+1)%opt['asd_batch_size'] == 0 or n_iter+1==len(train_loader):
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
                    scores = model(data[:,st:ed,:].cuda())
                    
                    scores = torch.mul(scores, loss_masks[:,st:ed].cuda())
                    loss = loss_function(labels[:,st:ed,:],scores,opt)
                    cost = loss["cost"] 
                    
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                    
                    epoch_cost += loss["cost"].cpu().detach().numpy()
    
                data = []
                labels = []     
                loss_masks = []      

        writer.add_scalars('data/cost', {'train': epoch_cost/(n_iter+1)}, n_epoch)        
        print( "ASD training loss(epoch %d): start - %.03f" %(n_epoch,epoch_cost/(n_iter+1)))
        
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
        for n_iter,(input_data,label_action,label_start) in enumerate(test_loader):
            data.append(input_data)
            labels.append(label_start)
            label_size = label_start.size()
            loss_masks.append(torch.ones(label_size))
                        
            if (n_iter+1)%opt['asd_batch_size'] == 0 or n_iter+1==len(test_loader):
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
                    scores = model(data[:,st:ed,:].cuda())
                    scores = torch.mul(scores, loss_masks[:,st:ed].cuda())
                    
                    loss = loss_function(labels[:,st:ed,:],scores,opt)
                    cost = loss["cost"] 
                                        
                    epoch_cost += loss["cost"].cpu().detach().numpy()
    
                data = []
                labels = []     
                loss_masks = []      
            
        writer.add_scalars('data/cost', {'test': epoch_cost/(n_iter+1)}, n_epoch)
        print( "ASD testing  loss(epoch %d): start - %.03f" %(n_epoch,epoch_cost/(n_iter+1)))
          
        #################save checkpoint################                                                                              
        state = {'epoch': n_epoch + 1,
                    'state_dict': model.state_dict()}
        torch.save(state, opt["checkpoint_path"]+"/asd_checkpoint.pth.tar" )
        if epoch_cost< model.module.mhd_best_loss:
            model.module.mhd_best_loss = np.mean(epoch_cost)
            torch.save(state, opt["checkpoint_path"]+"/asd_best.pth.tar" )
        writer.close()
        
def test_ASD(opt): 
    mhd_model = MHD(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/mhd_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    mhd_model.load_state_dict(base_dict)
    #mhd_model = torch.nn.DataParallel(mhd_model, device_ids=[0]).cuda()
    mhd_model.eval()
    
    asd_model = ASD(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/asd_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    asd_model.load_state_dict(base_dict)
    #asd_model = torch.nn.DataParallel(asd_model, device_ids=[0]).cuda()
    asd_model.eval()
    
    edr_model = EDR(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"]+"/edr_best.pth.tar")
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    edr_model.load_state_dict(base_dict)
    #edr_model = torch.nn.DataParallel(edr_model, device_ids=[0]).cuda()
    edr_model.eval()
    
    dataset = VideoDataSet(opt,subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, shuffle=False,
                                                num_workers=0, pin_memory=True,drop_last=True)
                     
    result_dict={}
    
    start_time = time.time()
    total_frames =0 
    
    with torch.no_grad():
        #process videoes one-by-one 
        for n_iter,(index,input_data,video_feature_frame) in enumerate(test_loader):
            video_name = test_loader.dataset.video_list[index]
            video_fps = dataset.video_dict[video_name]["fps"]
            temporal_interval=opt['temporal_interval']
            print(str(n_iter)+': '+video_name)
            
            duration = input_data.size(1)
            total_frames += duration
            
            ###init
            proposal_dict=[]
            proposals = [[] for i in range(0, opt['num_of_class']-1)]
                 
            max_activ=[[0] for i in range(0, opt['num_of_class']-1)]
            edr_input_history=[]
                            
            mhd_h=[]
            mhd_c=[]        
            for i in range(0,2):
                mhd_h.append(torch.zeros(1, opt["mhd_hidden_dim"]).cuda())
                mhd_c.append(torch.zeros(1, opt["mhd_hidden_dim"]).cuda())
            score, ed, mhd_h, mhd_c = mhd_model.forward_infer(input_data[:,0,:].cuda(), mhd_h, mhd_c)
            for i in range(0,5):
                edr_input_history.append(torch.stack([ed.detach()[:,:-1],score.detach()[:,:-1]],dim=-1))
            
            #process frames
            for i in range(0, duration):
                ###score, ed <- multi-head detector
                score, ed, mhd_h, mhd_c = mhd_model.forward_infer(input_data[:,i,:].cuda(), mhd_h, mhd_c)
                score = torch.softmax(score, dim=-1)
                ed = torch.softmax(ed,dim=-1)
                score = score.detach()
                ed = ed.detach()
                class_list=[]
                         
                ###activ_prob <- action end refinement                  
                edr_input_data=torch.stack([ed[:,:-1],score[:,:-1]],dim=-1)
                edr_input_history.pop(0)
                edr_input_history.append(edr_input_data)
                
                sub_input=torch.cat(edr_input_history,dim=-1)
                sub_input=sub_input.reshape(1,-1)         
                activ_prob = edr_model(sub_input.cuda())
                activ_prob = torch.softmax(activ_prob,dim=-1)
                activ_prob = activ_prob.detach().cpu().numpy()[0]
                
                score = score.cpu().numpy()[0]
                ed = ed.cpu().numpy()[0]
                
                ###thresholding for activating backward pass
                
                for classidx in range(0, opt['num_of_class']-1):
                    if(activ_prob[classidx] > opt['forward_thres']):
                        if( len(max_activ[classidx])==3 and (i-max_activ[classidx][2])*temporal_interval/video_fps > opt['allowed_delay']):                        
                            class_list.append([classidx]+max_activ[classidx][1:])
                            max_activ[classidx]=[0]
                        if(max_activ[classidx][0] < activ_prob[classidx] ):
                            max_activ[classidx]=[activ_prob[classidx], ed[classidx], i]
                    elif(len(max_activ[classidx])==3):
                        class_list.append([classidx]+max_activ[classidx][1:])
                        max_activ[classidx]=[0]
               
                #backward pass        
                for idx, [classidx, ed_score, ed_idx] in enumerate(class_list):
                    ###init
                    asd_h=torch.zeros(1, opt["asd_hidden_dim"]).cuda()
                    asd_c=torch.zeros(1, opt["asd_hidden_dim"]).cuda()
                    st, asd_h, asd_c = asd_model.forward_infer(input_data[:,ed_idx,:].cuda(), asd_h, asd_c)
                    
                    ###process backward frames (max_back_len)
                    max_st=[0]                    
                    for j in range(ed_idx, max(ed_idx-opt["max_back_len"],0),-1):
                        st, asd_h, asd_c = asd_model.forward_infer(input_data[:,j,:].cuda(), asd_h, asd_c)
                        if(ed_idx-j<2):
                            continue
                        st = torch.softmax(st, dim=-1)
                        st = st.detach().cpu().numpy()[0]
                        
                        ###thresholding for confirming action boundary
                        if(st[classidx] > opt["backward_thres"]):
                            if(max_st[0] < st[classidx]):
                                max_st=[st[classidx], j]
                        
                        elif(len(max_st) == 2):
                            insert_flag = True
                            conf = max_st[0] * ed_score
                            
                            ###check IoU before inserting to final proposal list
                            for proposal in proposals[classidx]:
                                if( proposal[1] > max_st[1] ):
                                    inter_min = max(proposal[0], max_st[1])
                                    inter_max = min(proposal[1], ed_idx)
                                    union_min = min(proposal[0], max_st[1])
                                    union_max = max(proposal[1], ed_idx)
                                    iou = float(inter_max - inter_min) / (union_max - union_min)
                                    
                                    if(iou > opt['iou_alpha']):
                                        insert_flag = False
                            if(insert_flag):
                                proposals[classidx].append([max_st[1],ed_idx, max_st[0], ed_score, conf])
                            break
            
            ###convert frames to time(sec) / formatting to ActivityNet eval API                
            for classidx in range(0, opt['num_of_class']-1):
                for proposal in proposals[classidx]:
                    tmp_dict = {}
                    tmp_dict["score"]= proposal[4]*1.0
                    tmp_dict["segment"] = [proposal[0]*temporal_interval/video_fps, proposal[1]*temporal_interval/video_fps]
                    tmp_dict["label"] = dataset.label_name[classidx]
                    proposal_dict.append(tmp_dict)
                    
            result_dict[video_name]=proposal_dict
            
    end_time = time.time()
    working_time = end_time-start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames/working_time, total_frames))
    
    ###dump & pass to ActivityNet eval API
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}}
    outfile=open(opt["result_file"],"w")
    json.dump(output_dict,outfile, indent=2)
    outfile.close()
    
    evaluation_detection(opt)
