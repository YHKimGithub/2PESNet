# -*- coding: utf-8 -*-
import sys
sys.path.append('./Evaluation')
from eval_detection import ANETdetection
import numpy as np

def run_evaluation_detection(ground_truth_filename, prediction_filename, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=True, check_status=False)
    anet_detection.evaluate()
    
    ap = anet_detection.ap
    mAP = anet_detection.mAP
    
    return (mAP, ap)

def evaluation_detection(opt):
    
    mAP, AP = run_evaluation_detection(
        opt["video_anno"],
        opt["result_file"],
        tiou_thresholds=np.linspace(0.3, 0.70, 5),
        subset=opt['inference_subset'])
        
    print('mAP')
    print(mAP)
    #print('AP')
    #print(AP)

