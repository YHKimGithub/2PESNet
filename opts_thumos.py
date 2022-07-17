import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--module',
        type=str)
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')        
    parser.add_argument(
        '--max_rnn_input',
        type=int,
        default=400)        
    
    # Dataset specific params (no need to change)
    parser.add_argument(
        '--num_of_class',
        type=int,
        default=21)   
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/thumos14_v2.json")        
    parser.add_argument(
        '--video_feature_rgb_train',
        type=str,
        default="./data/rgb_feature_valid_interval5.h5")
    parser.add_argument(
        '--video_feature_flow_train',
        type=str,
        default="./data/flow_feature_valid_interval5.h5")    
    parser.add_argument(
        '--video_feature_rgb_test',
        type=str,
        default="./data/rgb_feature_test_interval5.h5")
    parser.add_argument(
        '--video_feature_flow_test',
        type=str,
        default="./data/flow_feature_test_interval5.h5")
    parser.add_argument(
        '--temporal_interval',
        type=int,
        default=5)
    parser.add_argument(
        '--predefined_fps',
        type=int,
        default=1)
    parser.add_argument(
        '--boundary_ratio',
        type=float,
        default=0.01)
        
    #Multi Head Detector
    parser.add_argument(
        '--mhd_feat_dim',
        type=int,
        default=2048)
    parser.add_argument(
        '--mhd_hidden_dim',
        type=int,
        default=4096)
    parser.add_argument(
        '--mhd_out_dim',
        type=int,
        default=21)
        
    # MHD Training settings
    parser.add_argument(
        '--mhd_batch_size',
        type=int,
        default=8)
    parser.add_argument(
        '--mhd_training_lr',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--mhd_weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--mhd_epoch',
        type=int,
        default=60)
    parser.add_argument(
        '--mhd_step_size',
        type=int,
        default=20)
    parser.add_argument(
        '--mhd_step_gamma',
        type=float,
        default=0.1)
    
    #End Detection Refinement
    parser.add_argument(
        '--edr_input_dim',
        type=int,
        default=200)
    parser.add_argument(
        '--edr_hidden_dim',
        type=int,
        default=128)
    parser.add_argument(
        '--edr_out_dim',
        type=int,
        default=21)
        
    # EDR Training settings
    parser.add_argument(
        '--edr_input_score_size',
        type=int,
        default=5)
    parser.add_argument(
        '--edr_loss_foredist',
        type=float,
        default=0)
    parser.add_argument(
        '--edr_batch_size',
        type=int,
        default=1)
    parser.add_argument(
        '--edr_training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--edr_weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--edr_epoch',
        type=int,
        default=25)
    parser.add_argument(
        '--edr_step_gamma',
        type=float,
        default=0.1)
        
    #Action Start Detection
    parser.add_argument(
        '--asd_feat_dim',
        type=int,
        default=2048)
    parser.add_argument(
        '--asd_hidden_dim',
        type=int,
        default=4096)
    parser.add_argument(
        '--asd_out_dim',
        type=int,
        default=21)
        
    # ASD Training settings
    parser.add_argument(
        '--asd_batch_size',
        type=int,
        default=8)
    parser.add_argument(
        '--asd_training_lr',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--asd_weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--asd_epoch',
        type=int,
        default=60)
    parser.add_argument(
        '--asd_step_size',
        type=int,
        default=20)
    parser.add_argument(
        '--asd_step_gamma',
        type=float,
        default=0.1)
        
    # Inference time params
    parser.add_argument(
        '--allowed_delay',
        type=float,
        default=0.0)
    parser.add_argument(
        '--max_back_len',
        type=int,
        default=100)
    parser.add_argument(
        '--forward_thres',
        type=float,
        default=0.1)
    parser.add_argument(
        '--backward_thres',
        type=float,
        default=0.2)
    parser.add_argument(
        '--inference_subset',
        type=str,
        default="test")
    parser.add_argument(
        '--iou_alpha',
        type=float,
        default=0.5)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    
    args = parser.parse_args()

    return args
