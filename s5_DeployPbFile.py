# -*- coding: utf-8 -*-
"""
Convert Model to .pb file which is used to deploy the model on server
"""
import os
import tensorflow as tf


def ModelToPB(model_path,pb_path):
    from ModelTools import RestoreModel, FreezeGraphToPB
    tf.reset_default_graph()
    sess = tf.Session()
    sess, x_img, y_label, keep_prob, training, output_prob, output_label, train_step = RestoreModel(sess,model_path)     
    FreezeGraphToPB(sess,pb_path,output_node=[output_label,output_prob])
    sess.close()
  

if __name__ == '__main__':   
   model_path = os.path.join(os.getcwd(),'OutputModel',
                                         'colab_201902100257',                                        
                                         'MyDataV2_m4')      
   pbfile_path = model_path+".pb"
   test_path = os.path.join(os.getcwd(),'InputData',
                         'MyDataV2_test.tfrecords') 
   # Model to .pb file  
   ModelToPB(model_path,pbfile_path)
   
   # Chk .pb file   
   from QuickFunctions import PredMyRealImgs, PredTestSet
   PredMyRealImgs(pbfile_path)

   # Chk .pb file with test set
   PredTestSet(pbfile_path,test_path,save=False)

   
   

   
   



