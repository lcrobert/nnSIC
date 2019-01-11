# -*- coding: utf-8 -*-
"""
Convert Model to .pb file which is used to deploy the model on server
"""
import os
import tensorflow as tf


def ModelToPB(model_path,pb_path):
    from ModelTools import RestoreModelVariable, FreezeGraphToPB
    tf.reset_default_graph()
    sess = tf.Session() 
    x_img, y_label, keep_prob, training, output_propb, output_label = RestoreModelVariable(sess,model_path)
    FreezeGraphToPB(sess,pb_path,output_node=[output_label,output_propb])
    sess.close()
  


if __name__ == '__main__':
   # Model to .pb file  
   model_path = os.path.join(os.getcwd(),'OutputModel',
                                         'colab_201901040210',                                        
                                         'MyDataV1_m2')     
   pbfile_path = model_path+"_test.pb"
   ModelToPB(model_path,pbfile_path)
   
   """# Test .pb file   
   from QuickFunctions import PredMyRealImgsPB
   PredMyRealImgsPB(pbfile_path)
   """
   
   """
   # Read .pb file and get tensor
   from ModelTools import LoadGraphFromPB
   tf.reset_default_graph()
   graph = LoadGraphFromPB(pbfile_path)
   x_img = graph.get_tensor_by_name('import/InputData/x_img:0')
   keep_prob =graph.get_tensor_by_name('import/dropout:0')
   output_propb = graph.get_tensor_by_name('import/Output-Classification-layer/output_softmax/Softmax:0')
   output_label = graph.get_tensor_by_name('import/ArgMax:0')
   """
   #sess = tf.Session()
   
   



