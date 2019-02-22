# -*- coding: utf-8 -*-
"""
DataPipline - step 3

Creat TFRecord file of trainning, validation and test set.

  A. Read input .csv data : (eg.Image_Preprocessed_list.csv)
      . Get image_path_list and label_list
      . Check the distribution of labels of input data   
      . Use function change_label() if you want merge data or redefine label
      . function : change_label((ori_label,new_label), label_list)

  B. Seperate data to training-set and test-set 
      . Tuneable variable : train_test_ratio
      . Shuffle data in-place  
      . Trim data to 2 parts
      . Check and Get the distribution of labels of training-set and test-set 

  C. Re-sampling training-set of unbalanced label distribution 
      . Tuneable variable : label_min_counts
      . Over-sampling if sample size is less than label_min_counts
      . The return value shuffled already
      
  D. Seperate training-set to training-set and validation-set
      . Tuneable variable : train_vali_ratio
      . Trim data to 2 parts
      . Check the distribution of labels of training-set, validation-set and test-set 
 
  E. Creat TFRecords file (the input file of training stage)  
     . Tuneable variable : train_path, vali_path, test_path, flip_v 
     . Use function : data_to_TFRecords(filepath,imgs,labels,flip_v=False) 
     . If flip_v=True : Random choice a half of images to flip TOP_BOTTOM to increase diversity of data set

  F. Test TFRecords file   
     . Use the imported function : 
       from ReadTFRecords import chk_TFRecords_file
       chk_TFRecords_file(train_path,batch_size=4,plot=True)

"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import collections

input_main_path = os.path.join(os.getcwd(),'InputData')   

def data_to_TFRecords(filepath,imgs,labels,flip_v=False): 
    
    #Bytes data to tf.train.Feature format
    def bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #Int64 data to tf.train.Feature format
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))        
    
    data_size = len(labels)    
    if flip_v: # random choice a half of images to flip TOP_BOTTOM
       rdch_flip_idx = np.random.choice(data_size,size=int(data_size*0.5),replace=False)

    #build TFRecordWriter
    tfr_writer = tf.python_io.TFRecordWriter(filepath)
    progress = 0 
    print('------------------------------')      
    print('TFRecords - Transform start...')   
    #read each img and label
    for img_name, label in zip(imgs,labels):
        img = Image.open(img_name)
        if flip_v:
           if progress in rdch_flip_idx : 
              img = img.transpose(Image.FLIP_TOP_BOTTOM)        
        file_name = img_name.replace('\\','/').split('/')[-1]
        xsize,ysize = img.size                
        progress += 1
        print ("\r"+'Progress : %d / %d '%(progress,data_size), end="\r")     
        #tobytes 
        img_bytes = img.tobytes()    
        #build Features of Example function
        example = tf.train.Example(features=tf.train.Features(feature={
                 'height': int64_feature(ysize),
                 'width': int64_feature(xsize),
                 'image': bytes_feature(img_bytes),
                 'label': int64_feature(label),
                 'filename':bytes_feature(file_name.encode('utf8')),
                 }))    
        tfr_writer.write(example.SerializeToString())
    tfr_writer.close()
    print('\nTFRecords - Transform Done!')


def counter_label(label_list,stage=''): 
    label_counts = collections.Counter(label_list)
    label_counts_sorted = sorted(label_counts.items())
    print ('Label counts : %s'%stage)
    for i in label_counts_sorted: print (i)
    return label_counts 


def change_label(label_mapping=[],label_list=[]):
    """MY NEW LABEL
     {0='0':['led'],
      1='1'+'4':['cfl','ccfl','fluorescent','fluorescence','mercury','hg'],
      2='2'+'3':['sun','sky','solar','star','stellar','incandescent','tungsten','halogen','quartz'],
      3='5':['neon','ne'],
      4='6':['helium','he'],   
      5='7':['h','hydrogen'],
      6='8':['nitrogen'],           
      }
    """    
    #label_mapping = [(3,2),(4,1),(5,3),(6,4),(7,5),(8,6)] 
    new_label_list = np.copy(label_list)
    for lm in label_mapping: #(ori,new)
        new_label_list[ label_list==lm[0] ] = lm[1]         
    return new_label_list 


def shuffle_data(data_imgs,data_labels): #ndarray shuffle in-place
    shuffle_temp = np.array([data_imgs,data_labels]).T
    np.random.shuffle(shuffle_temp)
    np.random.shuffle(shuffle_temp) # shuffle twice
    data_imgs = shuffle_temp[:,0]
    data_labels = shuffle_temp[:,1]
    return data_imgs, data_labels


def trim_data_2parts(p1_ratio,data_imgs,data_labels):
    trim_size = int(p1_ratio*len(data_labels))
    p1_images = data_imgs[0:trim_size]
    p1_labels = data_labels[0:trim_size]
    p2_images = data_imgs[trim_size:]
    p2_labels = data_labels[trim_size:]
    return p1_images,p1_labels,p2_images,p2_labels 
 
    
def re_sampling(tra_images,tra_labels,tra_label_counts,label_min_counts=400,label_max_counts=500):     
    for label, counts in tra_label_counts.items():
        if counts < label_min_counts: #over-sampling
           the_img = tra_images[tra_labels==label]         
           rdch_img = np.random.choice(the_img,size=label_min_counts-counts)       
           tra_images = np.append(tra_images,rdch_img)
           tra_labels = np.append(tra_labels,np.zeros(rdch_img.size,dtype=int)+label)
        #if counts > label_max_counts: #down-sampling
        #   the_img = tra_images[tra_labels==0]
        #   the_label = tra_labels[tra_labels==0]         
        #   rdch_img = np.random.choice(the_img,size=label_max_counts,replace=False)                
    # shuffle again
    tra_images,tra_labels = shuffle_data(tra_images,tra_labels)
    return tra_images,tra_labels


#########################################################################

if __name__ == "__main__":
    
  # Setting TFRecords file save path        
  train_path = os.path.join(input_main_path,'MyDataV5_train.tfrecords')   
  vali_path = os.path.join(input_main_path,'MyDataV5_vali.tfrecords')   
  test_path = os.path.join(input_main_path,'MyDataV5_test.tfrecords')   
        
  # Read input .csv data    
  input_csv_path= os.path.join(input_main_path,'Image_Preprocessed_list.csv')   
  my_data = pd.read_csv(input_csv_path)
  path_list = my_data['ImgPath'].values
  label_list = my_data['Label'].values   
  # Get and check the distribution of input labels 
  label_counts = counter_label(label_list,'input ori') 

  ###########################################################
  # Change label (ori_label,new_label)
  label_mapping = [(3,2),(4,1),(5,3),(6,4),(7,5),(8,6)]
  label_list = change_label(label_mapping,label_list)
  label_counts = counter_label(label_list,'input changed') 
  ###########################################################
    
  # Randoming data shuffle in-place  
  images, labels = shuffle_data(path_list,label_list)  
  # Seperate data to training-set and test-set  
  train_test_ratio = 0.8
  tra_images,tra_labels,test_images,test_labels = trim_data_2parts(train_test_ratio,images,labels)     
  # Check and get the distribution of training-set and test-set 
  tra_label_counts = counter_label(tra_labels,'tra_labels') 
  test_label_counts = counter_label(test_labels,'test_labels') 

  ###########################################################
  # Re-sampling training-set to match the label_min_counts
  label_min_counts=480    
  tra_images, tra_labels = re_sampling(tra_images,tra_labels,tra_label_counts,label_min_counts=label_min_counts)   
  ###########################################################

  # Seperate training-set to training-set and validation-set   
  train_vali_ratio = 0.8
  train_images,train_labels,vali_images,vali_labels = trim_data_2parts(train_vali_ratio,tra_images,tra_labels)     
  # Check the distribution of training-set validation-set and test-set  
  train_label_counts = counter_label(train_labels,'final train_labels') 
  vali_label_counts = counter_label(vali_labels,'final vali_labels') 
  test_label_counts = counter_label(test_labels,'final test_labels') 


  #############################################################################
  #############################################################################
  chk_point = input("Creat TFRecords file [y/n]?")
  if chk_point == 'y':
     data_to_TFRecords(train_path, train_images, train_labels, flip_v=True)
     data_to_TFRecords(vali_path, vali_images, vali_labels, flip_v=True)
     data_to_TFRecords(test_path, test_images, test_labels, flip_v=False)
     print ('------------------------------')
     chk_point = input("Check TFRecord files [y/n]?")
     if chk_point == 'y':
        from ReadTFRecords import ChkTFRecordsFile
        for filepath in [train_path,vali_path,test_path]:
            file = ChkTFRecordsFile(filepath,batch_size=4)
            data = file.get_data(show_info=True,plot=True)
     else:
        print ("Skip check TFRecord files") 
  else:
     print ("Skip creat TFRecords files") 
     chk_point = input("Do you want to get all data from another TFRecord file [y/n]?")
     if chk_point == 'y':         
        from ReadTFRecords import ChkTFRecordsFile
        fname = input("The TFRecords file name : \n")
        file_path = os.path.join(input_main_path,fname)   
        if os.path.exists(file_path) and fname != '' and ' ' not in fname:                                       
           try: 
              file = ChkTFRecordsFile(file_path)
              data = file.get_all_data()         
              print ("""
                     Use data['key'] to get data (key : img, label, filename, imgsize, datasize)
                     Use file.count_label(data['label']) to show label distribution 
                     Use file.plot_imgs(data['img'][m:n]) to plot images
                     """)
           except Exception as e:
               print (e)
        else:
           print ("The file dos not exist. \nSkip check another TFRecord file ")            
     else:
        print ("Skip check another TFRecord file") 
 
  








 







