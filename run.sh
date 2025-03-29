#For mini dataset use 
#--dataroot= "/media/ava/Data_CI/Datasets/nuscenes-mini/" and mini

#For full dataset use 
#--dataroot="/media/ava/Data_CI/Datasets/nuscenes-full/nuscenes/" and trainval

#For export / inference -> In onnx_export.py change line no 41

#Original
# python main.py eval_model_iou mini  --modelf="model.pt" --dataroot="/media/ava/Data_CI/Datasets/nuscenes-mini/"

#Shabari
python main2.py eval_model_iou mini  --modelf="model.pt" --dataroot="/media/ava/Data_CI/Datasets/nuscenes-mini/"