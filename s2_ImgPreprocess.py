# -*- coding: utf-8 -*-
"""
DataPipline - step 2

Image preprocess: 
    Select few classes of images you want
    Recapture spectra area and Resize images to same new shape    
    Save preprocessed images to 'InputData/Image_Preprocessed' folder 
    Creat 'Image_Preprocessed_list.csv' file in 'InputData' folder
     
The labels:
    {'0':['led'],
     '1':['cfl','ccfl','fluorescent','fluorescence'],
     '2':['sun','sky','solar','star','stellar'],
     '3':['incandescent','tungsten','halogen','quartz'],   
     '4':['mercury','hg'],
     '5':['neon','ne'],
     '6':['helium','he'],   
     '7':['h','hydrogen'],
     '8':['nitrogen'],
     '9':['laser'],
     '10':['ar','argon'],}
"""
import os
import pandas as pd
import ImgPreprocessSingle as ips


def ImgPreprocess(label_selected,
                  preprocessed_folder=os.path.join(os.getcwd(),'InputData','Image_Preprocessed'),
                  preprocessed_result_path=os.path.join(os.getcwd(),'InputData','Image_Preprocessed_list.csv')):

    rawimg_folder = os.path.join(os.getcwd(),'RawData','Image_Labeled')
    
    if not os.path.exists(preprocessed_folder):os.makedirs(preprocessed_folder)     
    for label_name in set(label_selected): 
        label_folder = os.path.join(preprocessed_folder,label_name)       
        if not os.path.exists(label_folder): os.makedirs(label_folder)
            
    # Get img_paths and img_labels from rawimg_folder by label_selected  
    imgs_path = []
    imgs_label = []
    for label_name in label_selected: 
        folder_path = os.path.join(rawimg_folder,label_name)
        for item in os.listdir(folder_path):
            imgs_path.append(os.path.join(folder_path,item))
            imgs_label.append(int(label_name))
        
    ###################################################################
    #imgs_path = imgs_path[458:459]               
    prep_img_list = []
    prep_label_list = []
    skip_img = []
    data_size = len(imgs_path) 
    for i in range(data_size):   
        path = imgs_path[i]
        img_name = path.replace('\\','/').split('/')[-1] 
        print ("\r"+"Progress : %d / %d   "%(i+1,data_size),end="\r")    
        ##################################### 
        img_savepath = os.path.join(preprocessed_folder,str(imgs_label[i]),img_name)    
        try:
           img = ips.img_preprocess(imgpath=path, savepath=img_savepath, new_shape=(480,15), auto_flip=False, plot=False)    
           if img is '': 
              raise ValueError(img_name)
           else:
              prep_img_list.append(img_savepath)
              prep_label_list.append(imgs_label[i])
        except Exception as e:
           print('') 
           print('Image %d : '%(i+1), e, ' - skip')
           print('')
           skip_img.append(e)
    print ('\nDone!')                                
    print ('-------------------------------------------')                        
        
    my_data_prep_ok = pd.DataFrame({
                      "ImgPath": prep_img_list,  
                      "Label": prep_label_list,      
                      })         
    my_data_prep_ok.to_csv(preprocessed_result_path,sep=',')  

    return preprocessed_result_path  
        

    
if __name__ == '__main__':        
   # Select labels you want and use default path   
   label_selected = ["0","1","2","3","4","5","6","7","8"]
   preprocessed_result_path = ImgPreprocess(label_selected)
                
   # Or use custom path   
   #preprocessed_folder = os.path.join(os.getcwd(),'InputData','Image_Preprocessed_v2')
   #preprocessed_result_path = os.path.join(os.getcwd(),'InputData','Image_Preprocessed_list_v2.csv')            
   #preprocessed_result_path = ImgPreprocess(label_selected,preprocessed_folder,preprocessed_result_path)



   














