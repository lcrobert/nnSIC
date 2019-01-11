# -*- coding: utf-8 -*-
"""
Training Model
 * The all models were defined in CnnModelClass.py 
 * The dataset input pipeline is defined in ReadTFRecords.py 
 * - 1. Set static parameters
   - 2. set hyper parameters
   - 3. Set iteration steps
   - 4. Run training process
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from datetime import datetime

def RunTraining(iter_number,earlystop=True,evalValiConfusion=False): 
    from ReadTFRecords import InputDataPipeline as IDP                
    from CnnModelClass import ClassifierM1  
    
    sess = tf.Session()            
    data = IDP(sess,inputdata_path,batch_size,image_shape,label_size,epoch=0)    
    model = ClassifierM1(sess) 
    model.init_model(image_shape, label_size, convolution_kernel_size,
                     fc_layer_neuro_size, learn_rate, save_main_path)
                         
    # Settig early_stop param (acc based)
    no_improvement_steps = int(iter_number/3) 
    current_iterations = 0 
    last_improvement_step = 0    
    best_vali_accuracy = 0.0     
    # Init list of recording acc loss value (plot used) 
    record_idx = [] 
    train_acc_val = []        
    vali_acc_val = []   
    train_loss_val = []        
    vali_loss_val = []

    steps_per_epoch = round(data_size['train']/batch_size['train']) 
    StartTime = datetime.now()
    for ic in range(iter_number):         
        # Optimize 
        current_iterations += 1
        image_batch, label_batch = data.feeder(data.train_iterator)
        model.sess.run(model.train_step, 
                       feed_dict={model.x_img: image_batch*(1./255), 
                                  model.y_label: label_batch, 
                                  model.keep_prob: dropout,
                                  model.training: True})    

        # Calculate and Record acc loss of train and vali
        if current_iterations % 10 == 0 or current_iterations == iter_number:
           summary, train_loss, train_accuracy = model.sess.run([model.summary_merged_train, model.loss, model.accuracy],
                                              feed_dict={model.x_img: image_batch*(1./255), 
                                                         model.y_label: label_batch,
                                                         model.keep_prob: 1.0})
    
           image_batch, label_batch = data.feeder(data.vali_iterator) # only acc loss in summary_merged_vali        
           summary_vali, vali_loss, vali_accuracy = model.sess.run([model.summary_merged_vali, model.loss, model.accuracy],
                                              feed_dict={model.x_img: image_batch*(1./255), 
                                                         model.y_label: label_batch,
                                                         model.keep_prob: 1.0})                
           record_idx.append(current_iterations)            
           train_acc_val.append(train_accuracy)       
           vali_acc_val.append(vali_accuracy)           
           train_loss_val.append(train_loss)       
           vali_loss_val.append(vali_loss)                 
           model.train_writer.add_summary(summary, current_iterations) 
           model.train_writer.add_summary(summary_vali, current_iterations) 

           print('Iter: %d  Train_Acc: %4.2f%%  Vali_Acc: %4.2f%% '%(current_iterations,train_accuracy*100,vali_accuracy*100))            
           print('Iter: %d  TrainLoss: %1.4f  ValiLoss: %1.4f '%(current_iterations,train_loss,vali_loss))            
                  
           # If current validation accuracy is an improvement over best-known
           # then reflash early_stop param and save model (saver_earlystop)
           if vali_accuracy > best_vali_accuracy:
              best_vali_accuracy = vali_accuracy                 
              last_improvement_step = current_iterations
              if earlystop: 
                 model.saver_earlystop.save(model.sess, model_earlystop_savepath)

        # Calculate vali confusion mtx in every 2 epoch if needed
        if current_iterations % (2*steps_per_epoch) == 0 and evalValiConfusion: 
           image_batch, label_batch = data.feeder(data.vali_iterator)        
           acc, confusion_mtx = model.sess.run([model.accuracy, model.confusion],
                                         feed_dict={model.x_img: image_batch*(1./255), 
                                                    model.y_label: label_batch,
                                                    model.keep_prob: 1.0})
           cm_fig = PlotConfusionMatrix(confusion_mtx, show=False, save=False)
           cm_summery = FigToSummary(cm_fig,tag='ValiSet/ConfusionImage')
           model.train_writer.add_summary(cm_summery, current_iterations)
    
        # Early-Stopping  
        if earlystop and current_iterations - last_improvement_step > no_improvement_steps:
           print("-------------------------------------------------------")
           print("No improvement found in a while, stopping optimization.")
           print("Stop at step %d"%(current_iterations))        
           print("-------------------------------------------------------")
           break

    EndTime = datetime.now()
    print("Training completed!")
    print('Time usage : %s '%str(EndTime-StartTime))     
    print("Final improvement step : %d Vali_accuracy %4.2f%%"%(last_improvement_step,best_vali_accuracy*100))                     
    model.train_writer.add_graph(model.sess.graph)                                      
    model.train_writer.close() 
    model.saver.save(model.sess, model_savepath)        
    print("-------------------------------------------------------")    
    RunTestSet(model,data,save=True)      
    model.sess.close()         
    print("-------------------------------------------------------")
    result = {'step':record_idx,
              'train_acc':train_acc_val,
              'vali_acc':vali_acc_val,
              'train_loss':train_loss_val,
              'vali_loss':vali_loss_val}        
    return result  

   
def PlotConfusionMatrix(cm, show=True, save=True, tick_name=''):
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
    if save:
       plt.savefig(os.path.join(save_main_path,'ConfusionMatrix.png'),bbox_inches='tight',dpi=300)          
    if show:
       plt.show()
    else:
       plt.close()
       return fig
   

def RunTestSet(model,data,save=True):     
    image_batch, label_batch = data.feeder(data.test_iterator)                
    test_accuracy, confusion_mtx = model.sess.run([model.accuracy,model.confusion],
                                             feed_dict={model.x_img: image_batch*(1./255), 
                                                        model.y_label: label_batch,
                                                        model.keep_prob: 1.0})              
    print ('Test-set Acc: %4.2f%%'%(test_accuracy*100))  
    print ('Test-set ConfusionMatrix: \n',confusion_mtx)
    if save:
       np.savetxt(os.path.join(save_main_path,'TestAcc%4.2f_ConfusionMatrix.csv'%(test_accuracy*100)),confusion_mtx, delimiter=',', fmt=str("%d"))  
    PlotConfusionMatrix(confusion_mtx,show=True,save=save)


def PlotAccLoss(result,save=False):
    steps_per_epoch = round(data_size['train']/batch_size['train']) 
    epochs = [s/steps_per_epoch for s in result['step']] 

    fig1, ax1 = plt.subplots(num='Acc')
    ax1.set_title('Acc',fontdict={'fontsize':22,'position':(0.5, 0.08)})
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Steps')    
    ax1.plot(result['step'],result['train_acc'],'r-', label='train_acc')
    ax1.plot(result['step'],result['vali_acc'],'b-', label='vali_acc')
    ax1.tick_params(axis='x', labelcolor='k',direction='in')   
    ax2 = ax1.twiny() # share same y axis
    ax2.set_xlabel('Epochs',color='g')
    ax2.plot(epochs,result['train_acc'],'r-', label='train_acc')
    ax2.plot(epochs,result['vali_acc'],'b-', label='vali_acc')    
    ax2.tick_params(axis='x', labelcolor='g',direction='in')    
    plt.ylim(0,1.1)
    plt.legend(loc='best')
    fig1.tight_layout()
    if save:
       plt.savefig(os.path.join(save_main_path,'Acc.png'),bbox_inches='tight',dpi=150)      
    plt.show()

    fig3, ax3 = plt.subplots(num='Loss')
    ax3.set_title('Loss',fontdict={'fontsize':22,'position':(0.5, 0.80)})
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Steps')    
    ax3.plot(result['step'],result['train_loss'],'r-', label='train_loss')
    ax3.plot(result['step'],result['vali_loss'],'b-', label='vali_loss')
    ax3.tick_params(axis='x', labelcolor='k',direction='in')   
    ax4 = ax3.twiny() 
    ax4.set_xlabel('Epochs',color='g')
    ax4.plot(epochs,result['train_loss'],'r-', label='train_loss')
    ax4.plot(epochs,result['vali_loss'],'b-', label='vali_loss')    
    ax4.tick_params(axis='x', labelcolor='g',direction='in')    
    plt.legend(loc='best')
    fig1.tight_layout()
    if save:
       plt.savefig(os.path.join(save_main_path,'Loss.png'),bbox_inches='tight',dpi=150)      
    plt.show()


def SaveAccLossValue(result):
    import pandas as pd
    my_data_result = pd.DataFrame({
                      "step": result['step'],
                      "train_acc": result['train_acc'],  
                      "vali_acc": result['vali_acc'],
                      "train_loss": result['train_loss'],  
                      "vali_loss": result['vali_loss'],                       
                      })         
    my_data_result.to_csv(os.path.join(save_main_path,'AccLoss.csv'),sep=',') 


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
    EndTime = datetime.now() 
    print ('Time usage : %s '%str(EndTime-StartTime))
    print('-------------------------------------------')
    return x_img, y_label, keep_prob, training, output_propb, output_label


###########################################################################
if __name__ == '__main__':
    # Setting static parameter    
    input_main_path = os.path.join(os.getcwd(),'InputData')  
    inputdata_path = {'train':os.path.join(input_main_path,'MyDataV2_train.tfrecords'),
                      'vali':os.path.join(input_main_path,'MyDataV2_vali.tfrecords'),
                      'test':os.path.join(input_main_path,'MyDataV2_test.tfrecords')}    
    save_main_path = os.path.join(os.getcwd(),'OutputModel',datetime.now().strftime("%Y%m%d%H%M"))             
    model_savepath = os.path.join(save_main_path,'MyDataV2_m1')
    model_earlystop_savepath = os.path.join(save_main_path,'MyDataV1_m1_earlystop')    
    if not os.path.exists(save_main_path): os.makedirs(save_main_path)
    from ReadTFRecords import counter_TFRecord_datasize as ctd
    data_size = {'train':ctd(inputdata_path['train']),'vali':ctd(inputdata_path['vali']),'test':ctd(inputdata_path['test'])}                                       
    #data_size = {'train':2644,'vali':294,'test':262}
    label_size = 7
    image_shape = [15,480]
        
    # Setting hyper parameter
    batch_size = {'train':180,'vali':data_size['vali'],'test':data_size['test']}
    learn_rate = 0.0001#0.0001 
    fc_layer_neuro_size = 128 #  
    convolution_kernel_size = 9 #
    dropout = 0.5
    
    ###########################################################################    
    ###########################################################################     
    
    # Setting iter. number      
    iter_number = 1000
    epoch = round(iter_number*(batch_size['train']/data_size['train']))  
    # Training 
    tf.reset_default_graph()
    result = RunTraining(iter_number,earlystop=True,evalValiConfusion=True)
    PlotAccLoss(result,save=True)
    SaveAccLossValue(result)
    
    ###########################################################################     
    ########################################################################### 
    
    # Restore model variable of earlystop 
    tf.reset_default_graph()
    sess = tf.Session()
    x_img, y_label, keep_prob, training, output_propb, output_label = RestoreModelVariable(sess,model_earlystop_savepath)

    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_propb, 1), tf.argmax(y_label, 1)), tf.float32))
    confusion = tf.confusion_matrix(labels=tf.argmax(y_label, 1), predictions=tf.argmax(output_propb, 1), num_classes=7)

    from ReadTFRecords import InputData
    data = InputData(sess, inputdata_path['test'], batch_size['test'], image_shape, label_size)                                                                                    
    image_batch, label_batch = data.feeder(data.the_iterator)                
    accuracy, confusion_mtx = sess.run([acc, confusion],
                                       feed_dict={x_img: image_batch*(1./255), 
                                                  y_label: label_batch,
                                                  keep_prob: 1.0})              
    print ('Acc: %4.2f%%'%(accuracy*100))  
    print ('ConfusionMatrix: \n',confusion_mtx)    
    PlotConfusionMatrix(confusion_mtx, show=True, save=False)
    sess.close()

    ###########################################################

    """
    # Test model with unknown spectrum image 
    from QuickFunctions import PredMyRealImgs
    modelpath = os.path.join(os.getcwd(),'OutputModel',
                                         'colab_201901040210',
                                         'MyDataV1_m2')    
    PredMyRealImgs(modelpath)
    """



























