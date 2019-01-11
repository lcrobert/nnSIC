# -*- coding: utf-8 -*-
"""
A tool of checking labeled image duplicate or not.
"""
import os
import numpy as np
from PIL import Image
from datetime import datetime

# Image source         
rawimg_folder = os.path.join(os.getcwd(),'RawData','Image_Labeled')

def chk_img_duplicate(label_selected,delete=False):
    for label_name in label_selected:
        folder_path = os.path.join(rawimg_folder,label_name)
        data_size = len(os.listdir(folder_path))
        print ('-----------------------------')
        print ('Label :',label_name)         
        same = []
        img_temp = {}
        idx = 0        
        start_time = datetime.now()                
        for item in os.listdir(folder_path):
            idx += 1
            print ("\r"+'Progress : %d / %d    '%(idx,data_size),end="\r")        
            path = os.path.join(folder_path,item)
            filename = path.replace('\\', '/').split('/')[-1]
            img = Image.open(path).convert('L')     
            pix = np.asarray(img).tobytes()            
            if pix not in img_temp:
               img_temp[pix] = filename 
               #print (idx,':',filename)
            else: 
               same.append((filename,img_temp[pix]))
               #print (idx,':',filename,' same ')
        end_time = datetime.now()
        print ('\nTotal time usage : %s '%str(end_time-start_time))
        print ('Duplicate:')
        for item in same:
            if delete:
               os.remove(os.path.join(folder_path,item[0]))
               print (item,'rm:',item[0]) 
            else:
               print (item) 


"""# method 0
# Get img_paths and img_labels from rawimg_folder by label_selected  
label_selected = ["0","1","2","3","4","5","6","7","8"]   
label_selected = ["0"]             
imgs_path = []
imgs_label = []
for label_name in label_selected: 
    folder_path = os.path.join(rawimg_folder,label_name)
    for item in os.listdir(folder_path):
        imgs_path.append(os.path.join(folder_path,item))
        imgs_label.append(int(label_name))
###############################################
same = []
img_temp = {}
idx = 0
start_time = datetime.now()
for path in set(imgs_path):
    filename = path.replace('\\', '/').split('/')[-1]
    img = Image.open(path).convert('L')     
    pix = np.asarray(img).tobytes()
    idx += 1
    if pix not in img_temp:
       img_temp[pix] = filename 
       print (idx,':',filename)
    else: 
       same.append((filename,img_temp[pix]))
       print (idx,':',filename,' same ')
       #print (pix)       
end_time = datetime.now()
print ('Total time usage : %s '%str(end_time-start_time))
print (label_selected,':')
for item in same: print (item) 
"""


"""#another method 1
done = []
same = []
start_time = datetime.now()
for path1 in set(imgs_path):
    img1 = Image.open(path1)    
    x1, y1 = img1.size
    if img1.mode == 'RGBA': img1 = img1.convert('RGB')     
    pix1 = np.asarray(img1)    
    done.append(path1)
    inloop_time = datetime.now()
    runtimes = 0
    for path2 in set(imgs_path):
        if path2 not in done:
           runtimes += 1 
           img2 = Image.open(path2)    
           x2, y2 = img2.size
           if img2.mode == 'RGBA': img2 = img2.convert('RGB')                
           pix2 = np.asarray(img2)
           if x1==x2 and y1==y2:#same size
              if np.abs(np.mean(pix1-pix2)) < 5 :                  
                 same.append((path1.replace('\\', '/').split('/')[-1],path2.replace('\\', '/').split('/')[-1]))
    outloop_time = datetime.now()   
    print ('%d : %d loops  %s  %d '%(len(done),runtimes,str(outloop_time-inloop_time),len(same)))

end_time = datetime.now()
print ('Total time usage : %s '%str(end_time-start_time))
print (label_selected,':')
print (same) 
"""             



if __name__ == '__main__':
   label_selected = ["0","1","2","3","4","5","6","7","8","9","10"] 
   #label_selected = ["8"]     
   chk_img_duplicate(label_selected, delete=False)



        
        
        
    



   






























