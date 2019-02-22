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

       

def PlotConfusionMatrix(cm, show=True, savepath='',tick_name='', cmap='copper', colab=False):
    label_size = cm.shape[0]
    
    fig, ax = plt.subplots()   
    cm_img = ax.matshow(cm,cmap=cmap)#copper is better or 'coolwarm'
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
    if colab: ax.grid(which='major', visible=False) # Setting only for using CoLab    
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

def FigToSummary(fig, tag): 
    """
    Adapted from https://github.com/wookayin/tensorflow-plot/blob/master/tfplot/figure.py
    Convert a matplotlib figure ``fig`` into a TensorFlow Summary object
    that can be directly fed into ``Summary.FileWriter``.
      >>> fig, ax = ...
      >>> summary = Fig2summary(fig, tag='MyFigure/image')
      >>> summary_writer.add_summary(summary, global_step)
    """
    from io import BytesIO
    from tensorflow import Summary
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # get PNG data from the figure
    png_buffer = BytesIO()
    fig.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()    
    summary_image = Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                  encoded_image_string=png_encoded)
    summary = Summary(value=[Summary.Value(tag=tag, image=summary_image)])
    return summary



def RestoreModel(sess, model_path, custom=False):
    """Based on CnnModelClass.Classifier.build_saver()
    """
    print('-------------------------------------------')
    print('Restoring Model Variables...') 
    StartTime = datetime.now()
    # Use import_meta_graph to load meta data
    saver = tf.train.import_meta_graph(model_path+'.meta')  
    saver.restore(sess, model_path)
    if custom:
       EndTime = datetime.now() 
       print ('Time usage : %s '%str(EndTime-StartTime))
       print('-------------------------------------------')        
       return sess, tf.get_default_graph()
    else:
       x_img = tf.get_collection("InputImg")[0] #placeholder
       y_label = tf.get_collection("Inputlabel")[0] #placeholder
       keep_prob = tf.get_collection("Dropout")[0] #placeholder
       training = tf.get_collection("isTraining")[0] #placeholder 
       output_prob = tf.get_collection("OutputProb")[0]
       output_label = tf.get_collection("OutputLabel")[0]
       train_step = tf.get_collection("TrainSteps")[0]  
       EndTime = datetime.now() 
       print ('Time usage : %s '%str(EndTime-StartTime))
       print('-------------------------------------------')
       return sess, x_img, y_label, keep_prob, training, output_prob, output_label, train_step


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


def ShowPbInfo(pbfile_path, kwarg=''):
    tf.reset_default_graph()    
    graph = LoadGraphFromPB(pbfile_path)
    for op in graph.get_operations():
        if kwarg == '': 
           print (op.name)
        else:
           if kwarg in op.name:
              print (op) 


# For the model generated before 20190206
class OnlineModel(object):
    import ImgPreprocessSingle as ips 
    
    def __init__(self,mdpath):
        self.mdpath = [mdpath]  
        self.init_imgpath = os.path.join(os.getcwd(),'spapi','static','tf_models','init.png')        

        print('-------------------------------------------')    
        print('Online-Model init...')        
        StartTime = datetime.now()        
        self.RestorePbModel(self.mdpath[0]) 
        _, _ = self.pred(self.init_imgpath)
        EndTime = datetime.now() 
        print ('Time usage : %s '%str(EndTime-StartTime))
        print('-------------------------------------------')
        
    def RestorePbModel(self, pbfile_path):        
        tf.reset_default_graph()
        self.graph = LoadGraphFromPB(pbfile_path)
        self.x_img = self.graph.get_tensor_by_name('import/InputData/x_img:0')
        self.keep_prob = self.graph.get_tensor_by_name('import/dropout:0')
        self.output_prob = self.graph.get_tensor_by_name('import/Output-Classification-layer/output_softmax/Softmax:0')
        self.output_label = self.graph.get_tensor_by_name('import/ArgMax:0')
        self.sess = tf.Session()        
        
    def pred(self, img_path, print_result=False):
        path = img_path  
        img = self.ips.img_preprocess(imgpath=path, savepath='', new_shape=(480,15), auto_flip=False, plot=False)    
        
        if img is not '': 
           image_shape = img.shape 
           img = img.reshape((-1, image_shape[0], image_shape[1], 3))           
           label, prob = self.sess.run([self.output_label,self.output_prob], 
                                   feed_dict = {self.x_img: img*(1./255), 
                                                self.keep_prob: 1.0})  
           if print_result:             
              print('PrediLabel  =', label[0])
              print('Probability =',['%.2f'%(p) for p in prob[0]])
           return label[0], prob[0]
        else:        
           return None, None

#For the model generated after 20190207
class OnlineModelNew(OnlineModel):
    
    def RestorePbModel_new(self, pbfile_path):
        tf.reset_default_graph()    
        self.graph = LoadGraphFromPB(pbfile_path)
        self.x_img = self.graph.get_tensor_by_name('import/InputData/x_img:0')
        self.keep_prob = self.graph.get_tensor_by_name('import/dropout:0') 
        self.output_prob = self.graph.get_tensor_by_name('import/OutputClassification_layer/softmax/Softmax:0')
        self.output_label = self.graph.get_tensor_by_name('import/ArgMax_4:0')    
        self.sess = tf.Session() 


def ConfusionMatrixAnalysis(cm,f1_beta=1):
    """
    calculate
      precision, recall, f1, support, specificity, weighted_avg
    plot
     ROC: recall vs 1-specificity    
    """
    label_size = len(cm[0])
    cm_total = np.sum(cm)
    cm_sum_diagonal = np.sum(np.diagonal(cm))
    cm_acc = cm_sum_diagonal/cm_total
    
    precision = np.asarray([ (cm[i,i] if cm[i,i] != 0 else 1e-7  ) / ( np.sum(cm[:,i]) if np.sum(cm[:,i]) != 0 else 1 ) \
                            for i in range(label_size) ])
    
    recall = np.asarray([ cm[i,i]/np.sum(cm[i,:]) for i in range(label_size) ])
    f1_score = (1+f1_beta**2)*(precision*recall/(f1_beta**2*precision+recall))#beta=1
    support = np.sum(cm,axis=1)
    # specificity = FPR = TN/(TN+FP) # 在其他人中也預測為其他人的數量 / 自己以外(其他人)的true label數量
    specificity = np.asarray([(np.sum(cm)-np.sum(cm[:,i])-np.sum(cm[i,:])+np.sum(cm[i,i]))/(np.sum(cm)-np.sum(cm[i,:])) for i in range(label_size)])
    # weighted_avg = sum(P*N) / sum(N) 
    weighted_precision = np.sum(precision*support)/np.sum(support)
    weighted_recall = np.sum(recall*support)/np.sum(support)
    weighted_f1_score = np.sum(f1_score*support)/np.sum(support)

    # micro_avg-precision = sum(TP) / sum(TP+FP) for each label 
    # sum(TP) = cm_sum_diagonal
    # sum(TP+FP) = sum ( sum(cm[:,i]) ) = cm_total
    #micro_precision = cm_sum_diagonal/cm_total #equal to cm_acc 

    # micro_avg-recall = sum(TP) / sum(TN+TP) for each label
    # sum(TN+TP) = sum ( sum(cm[i,:]) ) = cm_total
    #micro_recall = cm_sum_diagonal/cm_total 

    # macro_avg = not useful for imbalance data   
    #macro_score = [np.mean(precision), np.mean(recall), np.mean(f1_score)]


    report = ""
    report += "Total Acc : %.4f \n    "%(cm_acc)
    report += "   ".join(["precision","   recall"," f1_score"," support"]) + "\n"
    for i in range(label_size):
      report += "%2d     "%(i)    
      report += "      ".join([ "%.4f"%v for v in [precision[i],recall[i],f1_score[i]]])        
      report += "      %5d"%(support[i])+"\n"     
    report += "\nw_avg"
    report += "  " + "%.4f      %.4f      %.4f"%(weighted_precision,weighted_recall,weighted_f1_score)
    report +="      %5d \n"%(np.sum(support))

    result = {"report":report,
              "weighted_avg":[weighted_precision,weighted_recall,weighted_f1_score],
              "precision":precision,
              "recall":recall,
              "f1_score":f1_score,
              "support":support,
              "specificity":specificity}

    return result


if __name__ == '__main__':
    
   tf.reset_default_graph()
   sess = tf.Session() 
   modelpath = os.path.join(os.getcwd(),'OutputModel',
                                        'colab_201901040210',
                                        'MyDataV1_m2') 
   output_path = modelpath+".pb"

   sess.close()





























