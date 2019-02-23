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
   
###################################################################
from ModelTools import OnlineModel
mdApath = os.path.join(os.getcwd(),'spapi','static','tf_models',                      
                      'co_7_201901040210','MyDataV1_m2.pb') 
model = OnlineModel(mdApath)
####################################################################
    
def vali_uploadimg(path):
    try :
       if path.endswith(('.tiff','.tif')): return False
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
              try :                 
                label, prob = model.pred(filepath,print_result=True)
                if label == None: raise ValueError('False of img_preprocess')                                       
                prob_max = max(prob)
                if prob_max < 0.2143: label = 'unknown'
                recognize_result = """               
                <span style="text-align: center; margin:0px auto; color: green;">
                  Recognized Result</span><br>
                <div style="text-align: center;">                        
                  label = %s <br>
                  probability = %.3f
                </div>                   
                """%(str(label),prob_max)                                                
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


    
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










