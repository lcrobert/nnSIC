# -*- coding: utf-8 -*-
"""
As title.
"""
import os
import numpy as np
#import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
from datetime import datetime

def PlotConfusionMatrix(cm, show=True, savepath='', tick_name=''):
    label_size = cm.shape[0]
    
    fig, ax = plt.subplots()   
    cm_img = ax.matshow(cm,cmap='copper')#copper is better
    fig.colorbar(cm_img)
    ax.set_title('ConfusionMatrix',fontdict={'fontsize':15})
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted') 
    tick_loc = np.arange(label_size)
    if tick_name == '':
       plt.xticks(tick_loc, range(label_size))
       plt.yticks(tick_loc, range(label_size)) 
    else:
       plt.xticks(tick_loc, tick_name)
       plt.yticks(tick_loc, tick_name)        
    ax.tick_params(axis='x', labelcolor='r') 
    ax.tick_params(axis='y', labelcolor='r')
    # Turn fig boundary line off and create white grid.    
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(tick_loc[0:-1]+0.5, minor = True)
    ax.set_yticks(tick_loc[0:-1]+0.5, minor = True)        
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)    
    # Loop over data dimensions and create text annotations.
    for i in range(label_size):
        for j in range(label_size):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="w")
    plt.tight_layout()
    if savepath != '':
       plt.savefig(savepath,bbox_inches='tight',dpi=300)          
    if show:
       plt.show()
    else:
       plt.close()
       return fig


def RestoreModelVariable(sess, model_path):
    """Based on CnnModelClass.build_saver()
    """
    print('-------------------------------------------')
    print('Restoring Model Variables...') 
    StartTime = datetime.now()
    # Use import_meta_graph to load meta data
    saver = tf.train.import_meta_graph(model_path+'.meta')
    saver.restore(sess, model_path)    
    x_img = tf.get_collection("InputImg")[0] #placeholder
    y_label = tf.get_collection("Inputlabel")[0] #placeholder
    keep_prob = tf.get_collection("Dropout")[0] #placeholder
    training = tf.get_collection("isTraining")[0] #placeholder 
    output_propb = tf.get_collection("OutputPropb")[0]
    output_label = tf.get_collection("OutputLabel")[0]
    #graph = tf.get_default_graph() # find tensor name in graph   
    #for op in graph.get_operations(): 
    #   if 'Placeholder' in op.name: print(op.name)      
    #keep_prob = graph.get_tensor_by_name('Fully-Connected-Layer-2/Placeholder:0')
    EndTime = datetime.now() 
    print ('Time usage : %s '%str(EndTime-StartTime))
    print('-------------------------------------------')
    return x_img, y_label, keep_prob, training, output_propb, output_label


def RestoreModel(sess, model_path):
    """Based on CnnModelClass.build_saver()
    """
    print('-------------------------------------------')
    print('Restoring Model Variables...') 
    StartTime = datetime.now()
    # Use import_meta_graph to load meta data
    saver = tf.train.import_meta_graph(model_path+'.meta')
    saver.restore(sess, model_path)    
    x_img = tf.get_collection("InputImg")[0] #placeholder
    y_label = tf.get_collection("Inputlabel")[0] #placeholder
    keep_prob = tf.get_collection("Dropout")[0] #placeholder
    training = tf.get_collection("isTraining")[0] #placeholder 
    output_propb = tf.get_collection("OutputPropb")[0]
    output_label = tf.get_collection("OutputLabel")[0]
    train_step = tf.get_collection("TrainSteps")[0]    
    EndTime = datetime.now() 
    print ('Time usage : %s '%str(EndTime-StartTime))
    print('-------------------------------------------')
    return sess,x_img, y_label, keep_prob, training, output_propb, output_label, train_step


def FreezeGraphToPB(sess,output_path,output_node=None):
    if output_node == None: # Output 'all' nodes      
       output_node_names = [n.name for n in sess.graph_def.node]      
    else: # Output 'get_collection' nodes name
       output_node_names = [n.name.split(':')[0] for n in output_node ]
       # ex.'Output-Classification-layer/y_fc_softmax/Softmax:0'

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
           sess,sess.graph_def,output_node_names)

    # Save frozen .pb file method 1
    output_filename = output_path.replace('\\','/').split('/')[-1]
    output_path = output_path.replace(output_filename,'')
    tf.train.write_graph(frozen_graph_def, 
                         output_path, 
                         output_filename, 
                         as_text=False)
    """# Save frozen .pb file method 2
    with tf.gfile.GFile(output_filename, "wb") as f:  
        f.write(frozen_graph_def.SerializeToString())""" 
    #print (output_node_names)


def LoadGraphFromPB(pbfile_path): 
    # Read .pb file by tf.gfile.FastGFile  
    with tf.gfile.GFile(pbfile_path, 'rb') as f:  
        graph_def_pb = tf.GraphDef()  
        graph_def_pb.ParseFromString(f.read())    
        tf.import_graph_def(graph_def_pb)# using default name='import'         
        """ another choice
        input_X = tf.placeholder(tf.float32, [None, 15, 480, 3], name='x_img')
        y, y_prob = tf.import_graph_def(graph_def_pb,
                                        input_map={'Input/x_img:0': input_X},
                                        return_elements=["ArgMax:0", "Output-Classification-layer/y_fc_softmax/Softmax:0"])
        return input_X, y,  y_prob 
        """
    graph = tf.get_default_graph()   
    return graph


if __name__ == '__main__':
    
   tf.reset_default_graph()
   sess = tf.Session() 
   modelpath = os.path.join(os.getcwd(),'OutputModel',
                                        'colab_201901040210',
                                        'MyDataV1_m2') 
   output_path = modelpath+".pb"
   x_img, y_label, keep_prob, training, output_propb, output_label = RestoreModelVariable(sess,modelpath)    
   FreezeGraphToPB(sess,output_path,output_node=[output_label,output_propb])
   sess.close()

   """
   ###########################################################
   graph = LoadGraphFromPB(output_path)
   #for op in graph.get_operations():
   #    print(op.name)  
   x_img = graph.get_tensor_by_name('import/InputData/x_img:0')
   keep_prob =graph.get_tensor_by_name('import/dropout:0')
   training = graph.get_tensor_by_name('import/training:0') 
   output_propb = graph.get_tensor_by_name('import/Output-Classification-layer/output_softmax/Softmax:0')
   output_label = graph.get_tensor_by_name('import/ArgMax:0')
   ###########################################################
   """


























