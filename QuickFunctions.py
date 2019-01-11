# -*- coding: utf-8 -*-
"""
Integrated functions
"""
import os
import numpy as np
#import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt 
#from datetime import datetime

def PredMyRealImgs(modelpath,imgfolder=''): 
      
    label_def = {'0':['led'],
                 '1':['cfl','ccfl','fluorescent','fluorescence','mercury','hg'],
                 '2':['sun','sky','solar','star','stellar','incandescent','tungsten','halogen','quartz'],
                 '3':['neon','ne'],
                 '4':['helium','he'],   
                 '5':['hydrogen','h'],
                 '6':['nitrogen']}
    img_list = ['neon_21411.png', 'cfl_P_20180714_conti.jpg',                       
                'cfl_DSC5202.JPG','neon_IMG_9999_full.JPG',                       
                'neon_DSC_2624.JPG','cfl_2936.png',                      
                'he_20s_corp.jpg','neon_IMG_20170118.jpg',                      
                'led_8973.jpg','led_3144553_n.jpg',                       
                'sky_80865_n.jpg','cfl_609251_n.jpg',                      
                'led_2448841_n.jpg','neon_DSC_2991.JPG',                       
                'sun_95084215_n.jpg','sun_97600_n.jpg',
                'he_8550.jpg','h_100897.png',
                'sun_sun.jpg','cfl_IMG_20170106.jpg']                                                                  
    #true_labels = [3,1,1,3,3,1,4,3,0,0,2,1,0,3,2,2,4,5,2,1]
    img_class = [n.split('_')[0] for n in img_list]
    true_labels = []
    for ic in img_class:
        for key, values in label_def.items():            
            if ic in values: true_labels.append(int(key)) 
        
    if imgfolder=='':                             
       mainpath = os.path.join(os.getcwd(),'RawData','MyRealImgs')  
    else:
        mainpath = imgfolder 
           
    # Restore model variable
    from ModelTools import RestoreModelVariable, PlotConfusionMatrix
    tf.reset_default_graph()
    sess = tf.Session()
    x_img, y_label, keep_prob, training, output_propb, output_label = RestoreModelVariable(sess,modelpath)
          
    import ImgPreprocessSingle as ips
    pred_labels = [] 
    for idx, imgname in enumerate(img_list):
        path = os.path.join(mainpath,imgname)   
        img = ips.img_preprocess(imgpath=path, savepath='', new_shape=(480,15), auto_flip=False, plot=False)    
        if img is not '': 
           image_shape = img.shape 
           img = img.reshape((-1, image_shape[0], image_shape[1], 3))           
           label, propb = sess.run([output_label,output_propb], 
                                   feed_dict = {x_img: img*(1./255), 
                                                keep_prob: 1.0})
           pred_labels.append(label[0])
           print(imgname+' :')
           print('TrueLabel   =', true_labels[idx])               
           print('PredLabel   =', label[0])
           print('Probability =',['%.2f'%(p) for p in propb[0]],'\n')
        else:
           print(imgname+' :','False of img_preprocess')               
    print ('True label : ', true_labels)
    print ('Pred label : ', pred_labels)    
    correct = np.equal(true_labels, pred_labels).astype(int)
    acc = np.mean(correct)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=true_labels,y_pred=pred_labels)
    tick_name = [value[0] for value in label_def.values() ]
    
    mn = modelpath.replace('\\','/').split('/')[-1]
    savepath = os.path.join(modelpath.replace(mn,''),'MyRealImgsConfusionMatrix_%4.2f.png'%(acc*100))
    PlotConfusionMatrix(cm, show=True, savepath=savepath,tick_name=tick_name)
    sess.close()
    print ('MyRealImgs Acc: %4.2f%%'%(acc*100)) 
    print('-------------------------------------------') 




def PredMyRealImgsPB(pb_path,imgfolder=''): 
      
    label_def = {'0':['led'],
                 '1':['cfl','ccfl','fluorescent','fluorescence','mercury','hg'],
                 '2':['sun','sky','solar','star','stellar','incandescent','tungsten','halogen','quartz'],
                 '3':['neon','ne'],
                 '4':['helium','he'],   
                 '5':['hydrogen','h'],
                 '6':['nitrogen']}
    img_list = ['neon_21411.png', 'cfl_P_20180714_conti.jpg',                       
                'cfl_DSC5202.JPG','neon_IMG_9999_full.JPG',                       
                'neon_DSC_2624.JPG','cfl_2936.png',                      
                'he_20s_corp.jpg','neon_IMG_20170118.jpg',                      
                'led_8973.jpg','led_3144553_n.jpg',                       
                'sky_80865_n.jpg','cfl_609251_n.jpg',                      
                'led_2448841_n.jpg','neon_DSC_2991.JPG',                       
                'sun_95084215_n.jpg','sun_97600_n.jpg',
                'he_8550.jpg','h_100897.png',
                'sun_sun.jpg','cfl_IMG_20170106.jpg']                                                                  
    #true_labels = [3,1,1,3,3,1,4,3,0,0,2,1,0,3,2,2,4,5,2,1]
    img_class = [n.split('_')[0] for n in img_list]
    true_labels = []
    for ic in img_class:
        for key, values in label_def.items():            
            if ic in values: true_labels.append(int(key)) 
        
    if imgfolder=='':                             
       mainpath = os.path.join(os.getcwd(),'RawData','MyRealImgs')  
    else:
        mainpath = imgfolder 
           
    
    # Restore model variable
    from ModelTools import LoadGraphFromPB, PlotConfusionMatrix
    tf.reset_default_graph()    
    graph = LoadGraphFromPB(pb_path)
    x_img = graph.get_tensor_by_name('import/InputData/x_img:0')
    keep_prob =graph.get_tensor_by_name('import/dropout:0')
    #training = graph.get_tensor_by_name('import/training:0') 
    output_propb = graph.get_tensor_by_name('import/Output-Classification-layer/output_softmax/Softmax:0')
    output_label = graph.get_tensor_by_name('import/ArgMax:0')    
                
    sess = tf.Session()
   
    import ImgPreprocessSingle as ips
    pred_labels = [] 
    for idx, imgname in enumerate(img_list):
        path = os.path.join(mainpath,imgname)   
        img = ips.img_preprocess(imgpath=path, savepath='', new_shape=(480,15), auto_flip=False, plot=False)    
        if img is not '': 
           image_shape = img.shape 
           img = img.reshape((-1, image_shape[0], image_shape[1], 3))           
           label, propb = sess.run([output_label,output_propb], 
                                   feed_dict = {x_img: img*(1./255), 
                                                keep_prob: 1.0})
           pred_labels.append(label[0])
           print(imgname+' :')
           print('TrueLabel   =', true_labels[idx])               
           print('PredLabel   =', label[0])
           print('Probability =',['%.2f'%(p) for p in propb[0]],'\n')
        else:
           print(imgname+' :','False of img_preprocess')               
    print ('True label : ', true_labels)
    print ('Pred label : ', pred_labels)    
    correct = np.equal(true_labels, pred_labels).astype(int)
    acc = np.mean(correct)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=true_labels,y_pred=pred_labels)
    tick_name = [value[0] for value in label_def.values() ]
    
    mn = pb_path.replace('\\','/').split('/')[-1]
    savepath = os.path.join(pb_path.replace(mn,''),'MyRealImgsConfusionMatrixPB_%4.2f.png'%(acc*100))
    PlotConfusionMatrix(cm, show=True, savepath=savepath,tick_name=tick_name)
    sess.close()
    print ('MyRealImgs Acc: %4.2f%%'%(acc*100)) 
    print('-------------------------------------------') 
    
    
    
    
if __name__ == '__main__':
   pass
   modelpath = os.path.join(os.getcwd(),'OutputModel',
                                        'colab_201901040210',
                                        'MyDataV1_m2')        
   pb_path = modelpath+".pb"     
   PredMyRealImgsPB(pb_path)    
    
    
    
    
    
    