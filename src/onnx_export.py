"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import onnxruntime as ort
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import numpy as np 
import onnx
from tqdm import tqdm

from .data import compile_data
from .tools import SimpleLoss,get_batch_iou
from .models2 import compile_model

def get_val_info(model, valloader, loss_fn, device, use_tqdm=True):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            
            print(f"Pytorch Prediction Shape {preds.shape}") #torch.Size([4, 1, 200, 200])
            # exit()
            
            onnx_filename= "lss-voxelpooling.onnx"
            
            dummy_input=(allimgs.cpu(),rots.cpu(),trans.cpu(),intrins.cpu(),post_rots.cpu(),post_trans.cpu())            

            
            # print("Converting model to ONNX...")
            # with torch.no_grad():
            #     torch.onnx.export(
            #             model.cpu(),  
            #             dummy_input,  
            #             onnx_filename,  
            #             opset_version=20,
            #             input_names=["allimgs","rots","trans","intrins","post_rots","post_trans"],  
            #             output_names=['output'],
            #             verbose=True
            #     )

            # print("Model successfully converted to ONNX and saved as", onnx_filename)
            ort_session = ort.InferenceSession("lss-voxelpooling.onnx")

            input_tensors = [allimgs.cpu(),rots.cpu(),trans.cpu(),intrins.cpu(),post_rots.cpu(),post_trans.cpu()]
            inputs =dict(zip([x.name for x in ort_session.get_inputs()], input_tensors))

            output_tensors = ort_session.run(None,inputs)
                
            exit()
            
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }


def eval_model_iou(version,
                modelf,
                dataroot='/data/nuscenes',
                gpuid=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    
    # Assuming you have a DataLoader object called 'dataloader'
    sample = next(iter(valloader))


    # print(len(sample))
    # exit()
    
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


