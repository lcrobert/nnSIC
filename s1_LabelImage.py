# -*- coding: utf-8 -*-
"""
DataPipline - step 1

A labeling tool of images downloaded by spectralworkbench_crawler
"""

import os
import re
import numpy as np
import pandas as pd
import collections

# Image source         
img_main_path = os.path.join(os.getcwd(),'RawData','Image')
# Image info source              
data_main_path = os.path.join(os.getcwd(),'RawData','DataTable')   

 
def SynDataTable_byImageExist(imgfolder_list,datatable_list):
    datatable_chk_list = np.asarray([ item[0:-17] for item in datatable_list])    
    all_data_size = 0
    new_all_data_size = 0
    for imgfolder in imgfolder_list:
        # find the datatable correspond to this image folder
        datatable_name = datatable_list[np.where(datatable_chk_list == imgfolder)]
        datatable_path = os.path.join(data_main_path, datatable_name[0]) 
        # current image list in this folder 
        img_id_list = np.asarray([the_id.split('.')[0] for the_id in os.listdir(os.path.join(img_main_path,imgfolder))], dtype=int)
        # read datatable via pd  
        data = pd.read_csv(datatable_path)
        # select data from datatable which in img_id_list
        new_data = data[data['ID'].isin(img_id_list)]#KEY
        # if data size is equal means there was no change of image status 
        if new_data.shape[0] != data.shape[0]: 
           new_data.to_csv(datatable_path, sep=',', index=False)    
        # record the change of data total size 
        all_data_size += data.shape[0]
        new_all_data_size += new_data.shape[0]        
        print ('----------------------------') 
        print ('ImageFolder : ', imgfolder) 
        print ('DataSize    : ', '%d >> %d'%(data.shape[0],new_data.shape[0])) 
    print ('--------------------------------') 
    print ('Orignal-data total size : %d'%(all_data_size))
    print ('New-data total size     : %d'%(new_all_data_size))
           
    
def count_datatable_keyword(datatable_list):
    keywords_all = np.array([],dtype=str)   
    # Read each datatable
    for table in datatable_list:
        table_path = os.path.join(data_main_path,table)              
        data = pd.read_csv(table_path)
        # Read each data point    
        for i in range(data.shape[0]):       
            keywords = re.sub("[' ]", "", data['keyword'][i][1:-1]).split(',')                   
            keywords_all = np.append(keywords_all,keywords)#used to count the keywords freq.                             
    return collections.Counter(keywords_all)


def get_label_by_keyword(datatable_list,keyword_group):
    keyword_filter = set(item for item in keyword_group.values() for item in item)#double for loop
    img_path_list = np.array([],dtype=str)
    img_label_list = np.array([],dtype=str)
    img_matched_keyword = np.array([],dtype=str)   
    # read each datatable 
    for table in datatable_list:
        table_path = os.path.join(data_main_path,table)              
        img_folder_path = os.path.join(img_main_path,table[0:-17])    
        data = pd.read_csv(table_path)
        # read each data point    
        for i in range(data.shape[0]): 
            # get keywords of data point
            keywords = re.sub("[' ]", "", data['keyword'][i][1:-1]).split(',')
            # get matched keywords
            keywords_match = list(set(keywords) & keyword_filter)   
            # find each matched_keyword belone to which label
            label_temp = []
            for word in keywords_match:
                for key, value_list in keyword_group.items():
                    if word in value_list: label_temp.append(key)
            # only one type of label        
            if len(set(label_temp)) == 1:
               img_label_list = np.append(img_label_list,label_temp[0])                                        
               img_matched_keyword = np.append(img_matched_keyword,keywords_match[0])  
               img_path_list = np.append(img_path_list,os.path.join(img_folder_path,data['ImageName'][i]))
            # multiple type of label   
            elif len(set(label_temp)) > 1: 
                 pass#print (data['ImageName'][i],' Ignored:multiple matched ',keywords_match)                             
            # no label matched  
            else:
                pass 
    print ('-------------------------------------------')            
    print ('Total pick-up data points : ',img_label_list.size)            
    return img_path_list, img_label_list, img_matched_keyword  


def img_to_folder_by_label(foldername,img_path_list,img_label_list):
    from shutil import copyfile
    path_list = img_path_list
    label_list = img_label_list
    imgfolder_path = os.path.join(os.getcwd(),'RawData',foldername)
    print ('-------------------------------------------')                
    # creat labeled-folder
    if not os.path.exists(imgfolder_path): 
       os.makedirs(imgfolder_path)
       for l in list(set(img_label_list)):
           os.makedirs(os.path.join(imgfolder_path,l)) 
    else: #remove all file and folder in imgfolder_path 
       print ('Remove existing file and folder...') 
       for root,dirs,files in os.walk(imgfolder_path):   
           if len(files) > 0:
              for item in files:           
                  os.remove(os.path.join(root,item)) 
              os.rmdir(root)
       print ('Remove Done!')     
       for l in list(set(img_label_list)):
           os.makedirs(os.path.join(imgfolder_path,l))           
    # copy img to labeled-folder    
    print ("Copy image to labeled-folder...") 
    img_newpath_list = []
    for idx in range(label_list.size):
        the_path = path_list[idx]
        the_label = label_list[idx]
        the_img_name = the_path.replace('\\','/').split('/')[-1]    
        new_path = os.path.join(imgfolder_path,str(the_label),the_img_name)    
        copyfile(the_path, new_path)
        print ("\r"+"progress : %d / %d  "%(idx+1,label_list.size),end="\r")
        img_newpath_list.append(new_path) 
    print ('Done!')                                
    print ('-------------------------------------------')                        
    return img_newpath_list   

    
####################################################################
keyword_group = {'0':['led'],
              '1':['cfl','ccfl','fluorescent','fluorescence'],
              '2':['sun','sky','solar','star','stellar'],
              '3':['incandescent','tungsten','halogen','quartz'],   
              '4':['mercury','hg'],
              '5':['neon','ne'],
              '6':['helium','he'],   
              '7':['h','hydrogen'],
              '8':['nitrogen'],
              '9':['laser'],
              '10':['ar','argon'],             
              }
####################################################################

if __name__ == '__main__':
   """
   First, manually check and delete bad images due to the extremely unstable 
   image quality of the source website. Then update the DataTable based 
   on the images exist or not. Run SynDataTable_byImageExist.
   """    
   #imgfolder_list = np.asarray(os.listdir(img_main_path))
   #datatable_list = np.asarray(os.listdir(data_main_path))   
   #SynDataTable_byImageExist(imgfolder_list,datatable_list)
   ####################################################################      
   #datatable_list = np.asarray(['12500-12301-201809180036.csv']) 
   datatable_list = np.asarray(os.listdir(data_main_path))
   #data_keyword_count = count_datatable_keyword(datatable_list)
   ####################################################################
   # Label images according to keyword_group
   img_path_list, img_label_list, img_matched_keyword = get_label_by_keyword(datatable_list,keyword_group)           
   #keyword_counter = collections.Counter(img_matched_keyword)
   #label_counter = collections.Counter(img_label_list)
   ####################################################################
   # Copy img to labeled-folder and get the path  
   img_labeledpath_list = img_to_folder_by_label('Image_Labeled_Temp',img_path_list,img_label_list)
   ####################################################################
   """
   Then manually check whether images in Image_Labeled_Temp are classified 
   correctly or has duplicates, and move or delete.
   Then manually merge into Image_Labeled directory. 
   """






