# -*- coding: utf-8 -*-
"""
A simple demo code of Django2.0+ view.py 
with corresponding templates/index.html file.
"""
from django.shortcuts import render
#from django.http import HttpResponse, JsonResponse
#from datetime import datetime
from django.core.files.storage import FileSystemStorage
from PIL import Image
import os
import tensorflow as tf
import ImgPreprocessSingle as ips
   
# Pre-load .pb file 
from ModelTools import LoadGraphFromPB
pbfile_path = os.path.join(os.getcwd(),'OutputModel',
                           'colab_201901040210',                                        
                           'MyDataV1_m2.pb')     
tf.reset_default_graph()
graph = LoadGraphFromPB(pbfile_path)
x_img = graph.get_tensor_by_name('import/InputData/x_img:0')
keep_prob =graph.get_tensor_by_name('import/dropout:0')
output_prob = graph.get_tensor_by_name('import/Output-Classification-layer/output_softmax/Softmax:0')
output_label = graph.get_tensor_by_name('import/ArgMax:0')
sess = tf.Session()


def pred(sess,img_path):
    path = img_path  
    img = ips.img_preprocess(imgpath=path, savepath='', new_shape=(480,15), auto_flip=False, plot=False)    
    if img is not '': 
       image_shape = img.shape 
       img = img.reshape((-1, image_shape[0], image_shape[1], 3))           
       label, prob = sess.run([output_label,output_prob], 
                               feed_dict = {x_img: img*(1./255), 
                                            keep_prob: 1.0})              
       print('PredLabel   =', label[0])
       print('Probability =',['%.2f'%(p) for p in prob[0]],'\n')
       return label[0], prob[0]
    else:        
       return None, None
    

def vali_uploadimg(path):
    # Check the uploaded file is img or not.
    try :	
       img = Image.open(path)
       img.close()
       return True
    except Exception as e:
       return False  
   

def Home(request): 
    upload_state = ''
    recognize_result = ''   
    descriptions = """
    <span style="text-align: left; color: black; font-size: 18px;">
        Descriptions of Seven Labels of Classifications : </span>
    <div style="margin-left:12px ; font-size: 16px;">                        
     label 0 : LED <br>
     label 1 : CCFL , Fluorescent Lamp, Mercury  <br>
     label 2 : Sun , Sky , Star , Incandescent Lamp , Halogen Lamp <br>
     label 3 : Neon Lamp <br> 
     label 4 : Helium Lamp <br>
     label 5 : Hydrogen Lamp <br>
     label 6 : Nitrogen Lamp     
    </div>    
    """            
    if request.method == 'POST':      
       if 'UPLOADimg' in request.POST and 'selected_img' in request.FILES: 
           myfile = request.FILES ['selected_img']
           fs = FileSystemStorage()
           filename = fs.save(myfile.name, myfile)
           filepath = fs.path(filename)
           if vali_uploadimg(filepath) == True : 
              upload_state = """
              <span style="font-size: 18px; text-align: center; margin:0px auto; color: green;">
               Image : %s </span>
              <div style=""> 
		           <img style="width: 480px; height: 150px;" src="%s" alt="image" />         
              </div>                                   
              """%(filename,fs.url(filename))
              
              # Run pred. process
              try :                 
                label, prob = pred(sess, filepath)
                if label == None: raise ValueError('False of img_preprocess')                    

                recognize_result = """               
                <span style="text-align: center; margin:0px auto; color: green;">
                  Recognized Result</span><br>
                <div style="text-align: center;">                        
                  label = %s <br>
                  probability = %.3f
                </div>                   
                """%(str(label),max(prob))                                                
              except Exception as e:
                print (e)
                recognize_result = """               
                <span style="text-align: center; margin:0px auto; color: green;">
                  Recognized Result</span><br>
                <div style="text-align: center; color: red;">                        
                  ERROR : %s
                </div>                   
                """%(e)                 

           else: 
              os.remove(filepath)
              upload_state = """
              <span style="font-size: 18px; text-align: center; margin:0px auto; color: red;">
              Please select a valid image file.</span>                   
              """                         
       else:           
           upload_state = """
           <span style="font-size: 18px; text-align: center; margin:0px auto; color: red;">
           Please select a image file.</span>                   
           """
    else:
        upload_state = ''
        recognize_result = ''
    return render(request, 'index.html',
                  {'upload_state': upload_state,
                   'recognize_result': recognize_result,
                   'descriptions' : descriptions,
                  })    


    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










