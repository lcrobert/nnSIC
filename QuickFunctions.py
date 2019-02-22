# -*- coding: utf-8 -*-
"""
Integrated functions for prediction
"""

import os
import numpy as np
#import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt 
#from datetime import datetime


class PredMyRealImgs(object):
    
      def init_images(self):
          self.label_def = {'0':['led'],
                         '1':['cfl','ccfl','fluorescent','fluorescence','mercury','hg'],
                         '2':['sun','sky','solar','star','stellar','incandescent','tungsten','halogen','quartz'],
                         '3':['neon','ne'],
                         '4':['helium','he'],   
                         '5':['hydrogen','h'],
                         '6':['nitrogen']}       
          self.img_list = ['neon_21411.png', 'cfl_P_20180714_conti.jpg',                       
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
          img_types = [n.split('_')[0] for n in self.img_list]
          self.true_labels = []
          for t in img_types:
              for key, values in self.label_def.items():            
                  if t in values: self.true_labels.append(int(key))    
              
      ##################################################################
      
      def __init__(self,model_path,imgfolder='',colab=False):  
          self.model_path = model_path
          self.model_name = model_path.replace("\\","/").split("/")[-1]
          self.init_images()
          self.colab = colab
		  
          if imgfolder=='':                             
             self.imgsfolder = os.path.join(os.getcwd(),'RawData','MyRealImgs')  
          else:
             self.imgsfolder = imgfolder                     

          if '.pb' in self.model_name: 
             self.restore_pb_model(model_path)
          else:
             self.restore_orignal_model(model_path)
             
          self.pred()
          self.cm_analysis()
          

      def restore_orignal_model(self, modelpath): 
          from ModelTools import RestoreModel
          tf.reset_default_graph()
          self.sess = tf.Session()
          self.sess, self.x_img, self.y_label, self.keep_prob, self.training, self.output_prob, self.output_label, self.train_step = RestoreModel(self.sess,modelpath)     
            
      def restore_pb_model(self, modelpath):
          from ModelTools import LoadGraphFromPB
          tf.reset_default_graph()    
          self.graph = LoadGraphFromPB(modelpath)
          self.x_img = self.graph.get_tensor_by_name('import/InputData/x_img:0')
          self.keep_prob = self.graph.get_tensor_by_name('import/dropout:0') 
          self.output_prob = self.graph.get_tensor_by_name('import/OutputClassification_layer/softmax/Softmax:0')
          self.output_label = self.graph.get_tensor_by_name('import/ArgMax_4:0')    
          #self.training = self.graph.get_tensor_by_name('import/training:0')                
          self.sess = tf.Session()          
          

      def pred(self):          
          import ImgPreprocessSingle as ips 
          self.pred_labels = [] 
          for idx, imgname in enumerate(self.img_list):
              path = os.path.join(self.imgsfolder,imgname)   
              img = ips.img_preprocess(imgpath=path, savepath='', new_shape=(480,15), auto_flip=False, plot=False)    
              if img is not '': 
                 image_shape = img.shape 
                 img = img.reshape((-1, image_shape[0], image_shape[1], 3))           
                 label, prob = self.sess.run([self.output_label, self.output_prob], 
                                        feed_dict = {self.x_img: img*(1./255), 
                                                     self.keep_prob: 1.0})
                 self.pred_labels.append(label[0])
                 print(imgname+' :')
                 print('TrueLabel   =', self.true_labels[idx])               
                 print('PredLabel   =', label[0])
                 print('Probability =',['%.2f'%(p) for p in prob[0]],'\n')
              else:
                 print(imgname+' :','False of img_preprocess') 
          correct = np.equal(self.true_labels, self.pred_labels).astype(int)
          self.acc = np.mean(correct)
          self.sess.close()                 
          print ('True label : ', self.true_labels)
          print ('Pred label : ', self.pred_labels) 
          print ('MyRealImgs Acc: %4.2f%%'%(self.acc*100)) 
          print('-------------------------------------------------')           

          
      def cm_analysis(self):
          from ModelTools import PlotConfusionMatrix, ConfusionMatrixAnalysis          
          from sklearn.metrics import confusion_matrix
          self.cm = confusion_matrix(y_true=self.true_labels,y_pred=self.pred_labels)
          tick_name = [ value[0] for value in self.label_def.values() ]
          if '.pb' in self.model_name:
             cm_img_name = 'MyRealImgsConfusionMatrixPB_%4.2f.png'%(self.acc*100) 
          else:
             cm_img_name = 'MyRealImgsConfusionMatrix_%4.2f.png'%(self.acc*100)
          savepath = os.path.join(self.model_path.replace(self.model_name,''),cm_img_name)
          PlotConfusionMatrix(self.cm, show=True, savepath=savepath,tick_name=tick_name, colab=self.colab)
          self.cm_result =  ConfusionMatrixAnalysis(self.cm)
          print (self.cm_result["report"])
          print('-------------------------------------------') 
    


class PredTestSet(PredMyRealImgs):
    
    
      def __init__(self,model_path,test_path,save=True,colab=False):  
          self.model_path = model_path
          self.model_name = model_path.replace("\\","/").split("/")[-1]
          self.testset_path = test_path
          testset_name = test_path.replace("\\","/").split("/")[-1]
          self.testset_name = testset_name.split(".")[0]
          self.colab = colab
		  
          if '.pb' in self.model_name: 
             self.restore_pb_model(model_path)
          else:
             self.restore_orignal_model(model_path)
             
          self.pred()
          self.cm_analysis(save)    

          
      def pred(self):
          from ReadTFRecords import InputData 
          self.data = InputData(self.sess, self.testset_path, batch_size=None, image_shape=[15,480], label_size=7)                                                                                    
          self.image_batch, self.label_batch = self.data.feeder(self.data.the_iterator)      
          acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output_prob, 1), tf.argmax(self.label_batch, 1)), tf.float32))
          cm = tf.confusion_matrix(labels=tf.argmax(self.label_batch, 1), predictions=tf.argmax(self.output_prob, 1), num_classes=7)
          self.acc, self.cm = self.sess.run([acc, cm],
                                    feed_dict={self.x_img: self.image_batch*(1./255), 
                                               self.keep_prob: 1.0})              
          print ('TestSet Acc: %4.2f%%'%(self.acc*100)) 
          print('-------------------------------------------------')           


      def cm_analysis(self,save):
          from ModelTools import PlotConfusionMatrix, ConfusionMatrixAnalysis          
          if '.pb' in self.model_name:
             cm_img_name = '_TestSetCM_PB_%4.2f.png'%(self.acc*100) 
          else:
             cm_img_name = '_TestSetCM_%4.2f.png'%(self.acc*100)
          savepath = os.path.join(self.model_path.replace(self.model_name,''),
                                  self.testset_name+cm_img_name)
          PlotConfusionMatrix(self.cm, show=True, savepath = savepath if save else '', colab=self.colab )
          self.cm_result =  ConfusionMatrixAnalysis(self.cm)
          print (self.cm_result["report"])
          print('-------------------------------------------') 
          self.sess.close() 
          

if __name__ == '__main__':

   modelpath = os.path.join(os.getcwd(),'OutputModel',
                                        'colab_201901040210',
                                        'MyDataV1_m2') 
   test_path = os.path.join(os.getcwd(),'InputData',
                            'MyDataV1_test.tfrecords')         
   pb_path = modelpath+".pb"     
   
   #PredMyRealImgs(pb_path)       
       
   #PredTestSet(pb_path,test_path)
    
    
    