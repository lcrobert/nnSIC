# -*- coding: utf-8 -*-
"""
Image preprocess tool of spectrum image.

Entry function : 
    img_preprocess(imgpath,savepath='',new_shape=(480,15),auto_flip=False, plot=False)        

"""
import os
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import matplotlib.pyplot as plt 
import cv2

###############################################
###############################################

from scipy.optimize import leastsq     
def gauss(x, p):
    noi = p[0]
    aa = p[1]
    mu = p[2]
    sigma =p[3] 
    fun = aa*np.exp(-(x-mu)**2/(2.*sigma**2))+noi
    return fun            
def residulas(p, fit_x, fit_y):
    return (gauss(fit_x, p)-fit_y)**2

#from scipy.optimize import curve_fit
#def fun(x,a,b):
#    return a*x+b 

###############################################
###############################################
    
def trim_loc_yaxis(img, ysize, xsize,trim_range=10):    
    max_vals_y = np.max(img, axis=1)#lens=ysize
            
    fit_x = np.arange(0,ysize)
    mean = np.mean(fit_x)
    std = np.std(fit_x)               
    amp = np.max(max_vals_y)
    pini = [5., amp, mean, std]    
    res, flag = leastsq(residulas, pini, args=(fit_x, max_vals_y),maxfev=100000)         
    fit_result = gauss(fit_x, res) #curve          
    fit_max_y = np.max(fit_result)
    fit_max_x = fit_x[fit_result == fit_max_y][0] # y_idx

    cut_y_up = fit_max_x-trim_range
    cut_y_down = fit_max_x+trim_range    
    if cut_y_up <= 0: 
       cut_y_up = 1
    if cut_y_down >= ysize: 
       cut_y_down = ysize-1
   ###############################################
    test_img =  img[cut_y_up:cut_y_down]       
    the_profile = test_img.sum(axis=0)/test_img.shape[0]
    sn = np.mean(the_profile)/np.std(the_profile)
#    print (np.mean(the_profile),np.std(the_profile))
#    print ('snr=',sn)    
    if sn < 1.3:
       #print('use sn') 
       cm_y,cm_x = ndimage.measurements.center_of_mass(img)
       max_vals_y = img[0:,int(cm_x)] 
       
       fit_x = np.arange(0,ysize)
       mean = np.mean(fit_x)
       std = np.std(fit_x)               
       amp = np.max(max_vals_y)
       pini = [5., amp, mean, std]    
       res, flag = leastsq(residulas, pini, args=(fit_x, max_vals_y),maxfev=100000)         
       fit_result = gauss(fit_x, res) #curve          
       fit_max_y = np.max(fit_result)
       fit_max_x = fit_x[fit_result == fit_max_y][0] # y_idx

       cut_y_up = fit_max_x-trim_range
       cut_y_down = fit_max_x+trim_range    
       if cut_y_up <= 0: 
          cut_y_up = 1
       if cut_y_down >= ysize: 
          cut_y_down = ysize-1

    return max_vals_y, fit_result, [fit_max_x,fit_max_y] , [cut_y_up,cut_y_down]
   
def trim_loc_left(b_profile, r_profile, xsize, g_profile):
    cutoff_point_left = int(np.argmax(b_profile)/3)
    for idx in range(cutoff_point_left,np.argmax(b_profile)):        
        curr_val = b_profile[idx]
        if idx+5 > xsize-1:
           next_5val = curr_val                
        else:    
           next_5val = b_profile[idx+5]         
        if b_profile[idx] > 1.0*r_profile[idx] and b_profile[idx] > g_profile[idx]:
            
            if curr_val < np.average(b_profile[idx:idx+4]):
               #is curve flat? 
               m = (next_5val-curr_val)/5
               deg = np.degrees(np.arctan(m))
               if np.abs(deg) > 18:
                  cutoff_point_left = idx
                  break                
            else: 
               pass            
    return cutoff_point_left
       
def trim_loc_right(b_profile, r_profile, xsize, g_profile):       
    inver_b_profile = b_profile[::-1]
    inver_r_profile = r_profile[::-1] 
    inver_g_profile = g_profile[::-1]
    right_idx = xsize-int(np.argmax(inver_r_profile)/6)
    for idx in range(1,np.argmax(inver_r_profile)):
        if inver_r_profile[idx] >inver_g_profile[idx] and inver_r_profile[idx] > inver_b_profile[idx]:
           curr_val = inver_r_profile[idx]
           if idx+5 > xsize-1:
              next_5val = curr_val                
           else:    
              next_5val = inver_r_profile[idx+5]   
           m = (next_5val-curr_val)/5
           deg = np.degrees(np.arctan(m))
           if np.abs(deg) > 18:
              right_idx = range(xsize)[-idx]
              break
           else: 
               pass           
    return right_idx 

###############################################
###############################################
    
def plot_ori_img(path):    
    img = Image.open(path)
    img_name = path.replace('\\','/').split('/')[-1]    
    cm = ndimage.measurements.center_of_mass(np.asarray(img))
    plt.figure('orignal image')
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)    
    plt.subplot(1,1,1) 
    plt.title('%s'%(img_name)) 
    plt.plot(cm[1],cm[0],'wo')
    plt.imshow(img)
    plt.tight_layout() 

def plot_trim_yaxis_result(img_name,max_vals_y,fit_result,trim_y_loc,top_point):
    fig1 = plt.figure('trim_Y-axis_result')
    ax = fig1.add_subplot(1,1,1)
    ax.set_title('%s'%(img_name)) 
    ax.set_ylim(0,255)
    ax.plot(max_vals_y)
    ax.plot(fit_result)
    ax.plot([trim_y_loc[0],trim_y_loc[0]],[30,200],'k--')    
    ax.plot([trim_y_loc[1],trim_y_loc[1]],[30,200],'y--')        
    ax.plot(top_point[0],top_point[1],'r^')    
    fig1.tight_layout()

def plot_trim_xaxis_result(img_name,b_profile,g_profile,r_profile,left_idx,right_idx):
    fig2 = plt.figure('trim_X-axis_result')
    ax = fig2.add_subplot(1,1,1)
    ax.set_title('%s'%(img_name)) 
    ax.plot(b_profile,'b-') 
    ax.plot(g_profile,'g-') 
    ax.plot(r_profile,'r-')
    ax.plot([left_idx,left_idx],[30,200],'k--')    
    ax.plot([right_idx,right_idx],[30,200],'y--')  
    fig2.tight_layout()
    
def plot_trimed_img(img_name,trim_img):
    fig3 = plt.figure('trimed-image')
    ax = fig3.add_subplot(1,1,1)
    ax.set_title('%s'%(img_name))
    ax.imshow(trim_img[:,:,::-1])  
    fig3.tight_layout()


def plot_trimed_img_profile(img_name,new_profile):
    fig4 = plt.figure('trimed-image profile')
    ax = fig4.add_subplot(1,1,1)
    ax.set_title('%s'%(img_name))
    ax.plot(new_profile,'g-')  
    fig4.tight_layout()
        
###############################################
############################################### 
def chk_flip(b_profile,g_profile,r_profile,xsize,cm_x):
    flip = False 
    max_g = np.argmax(g_profile)
    max_r = np.argmax(r_profile) 
    max_b = np.argmax(b_profile)
    w = (max_r+max_b+max_g)/3
    #print (max_r,max_g,max_b) #1361 2197 1361 r=b<g r<g=b in
    if (w < 30) or (w > xsize-30)  :
       st = int(30+cm_x/5)
       en = int(xsize-30-cm_x/5)
       max_g = np.argmax(g_profile[st:en])
       max_r = np.argmax(r_profile[st:en]) 
       max_b = np.argmax(b_profile[st:en])        
       #print (max_r,max_g,max_b)       
    left_b = b_profile[0:max_g].sum()
    left_r = r_profile[0:max_g].sum()
    right_b = b_profile[max_g:].sum()
    right_r = r_profile[max_g:].sum()         
    #print (left_b,left_r,right_r,right_r) 

    d_rb = abs(max_r-max_b)
    d_bg = abs(max_b-max_g)
    d_gr = abs(max_g-max_r)
    
    #print ((left_r > left_b and right_r < right_b))
    if (left_r > left_b and right_r < right_b) or \
       (left_r>right_r and left_b<right_b) or \
       (max_r < max_g < max_b) or \
       (max_r < max_g == max_b) or \
       (d_rb < 20 and max_r+20 < max_g) or \
       (d_rb < 20 and left_r > right_r or left_r > left_b ):       
       #print ("ENTER check")
       flip = False      
       if (left_r>right_r or left_b<right_b ):
          counts_L = 0    
          #left_blue_sn = np.max(b_profile[0:int(cm_x)])/np.std(b_profile[0:int(cm_x)])      
          left_blue_sn = 255/(np.std(b_profile[0:int(cm_x)]+10E-5))          
          blue_set_lo = b_profile[int(cm_x):]
          blue_set_l = blue_set_lo[blue_set_lo > left_blue_sn] 
          red_set_lo = r_profile[int(cm_x):]
          red_set_l = red_set_lo[blue_set_lo > left_blue_sn]        
          for ii in range(blue_set_l.size):
              if blue_set_l[ii] > red_set_l[ii]: counts_L += 1 
          #print ('left_blue_sn :',left_blue_sn)                   
          #print ('counts_L = ',counts/blue_set_l.size)          
          counts_R = 0
          #right_red_sn = np.max(r_profile[int(cm_x):])/np.std(r_profile[int(cm_x):]) 
          right_red_sn = 255/(np.std(r_profile[int(cm_x):])+10E-5) 
          red_set_ro = r_profile[int(cm_x):]
          red_set_r = red_set_ro[red_set_ro > right_red_sn] 
          blue_set_ro = b_profile[int(cm_x):]
          blue_set_r = blue_set_ro[red_set_ro > right_red_sn]    
          for ii in range(red_set_r.size):
              if red_set_r[ii] > blue_set_r[ii]: counts_R += 1 
          #print ('right_red_sn :',right_red_sn)
          #print ('counts_R = ',counts_R/red_set_r.size)          
          if counts_R > counts_L: 
             flip = False
          else: 
             flip = True
             #print ('s1-inverted')

       if (d_rb < 20 and left_r > right_r and left_r > left_b ):
          flip = True
          #print('s2-inverted')           

       if (max_r < max_g) or (max_r < max_b) or (max_g < max_b):       
          if (d_rb < 6 and d_bg < 6 and d_gr < 6) :
              if left_r < right_r:
                 flip = False
              else:
                 flip = True
                 #print('s3.1-inverted')             
          else:              
             flip = True
             #print('s3.2-inverted')    
             
    return flip
###############################################
###############################################
    
def img_preprocess(imgpath,savepath='',new_shape=(480,15),auto_flip=False, plot=False):        
    path = imgpath
    img_name = path.replace('\\','/').split('/')[-1] 
    try:
       img_cv2_c = cv2.imread(path, 1) # 0=mono 
       ysize, xsize, _ = img_cv2_c.shape
       if plot : plot_ori_img(path)
    except Exception as e:
       print(type(e).__name__," : ",e)
       return []   
    if ysize > xsize:  # Vertical image?
       img_cv2_c = ndimage.interpolation.rotate(img_cv2_c, angle=90.0)
       ysize, xsize, _ = img_cv2_c.shape
       #cv2.imshow("1",img_cv2_c)
    img_cv2 = cv2.cvtColor(img_cv2_c, cv2.COLOR_BGR2GRAY)
    #global cm_y,cm_x 
    cm_y,cm_x = ndimage.measurements.center_of_mass(img_cv2)
    #####################################     
    # get idx of trim up and down
    max_vals_y, fit_result, top_point, trim_y_loc = trim_loc_yaxis(img_cv2, ysize, xsize,trim_range=10)
    if plot : plot_trim_yaxis_result(img_name,max_vals_y,fit_result,trim_y_loc,top_point)
    #####################################     
    # flip img or not 
    b,g,r = cv2.split(img_cv2_c)    
    b_profile = b.sum(axis=0)/ysize
    g_profile = g.sum(axis=0)/ysize
    r_profile = r.sum(axis=0)/ysize
    if auto_flip:
       flip = chk_flip(b_profile,g_profile,r_profile,xsize,cm_x)
    else:        
       flip = False
    if flip:   
       img_cv2_c = cv2.flip(img_cv2_c, 1) # Flipped Horizontally
       b,g,r = cv2.split(img_cv2_c)    
       b_profile = b.sum(axis=0)/ysize
       g_profile = g.sum(axis=0)/ysize
       r_profile = r.sum(axis=0)/ysize                 
    #####################################              
    # get idx of trim left and right 
    trim_x_loc = [0,xsize-1]          
    trim_x_loc[0] = trim_loc_left(b_profile, r_profile, xsize, g_profile)
    trim_x_loc[1] = trim_loc_right(b_profile, r_profile, xsize, g_profile) 
    if trim_x_loc[0] >= trim_x_loc[1]: trim_x_loc = [5,xsize-5]#trim_x_loc[::-1]       
    if plot : plot_trim_xaxis_result(img_name,b_profile,g_profile,r_profile,trim_x_loc[0],trim_x_loc[1])    
    #####################################    
    # creat croped image and save to new path    
    try:       
       trim_img =  img_cv2_c[trim_y_loc[0]:trim_y_loc[1],trim_x_loc[0]:trim_x_loc[1]]
       #new_shape = (480,15)
       trim_img = cv2.resize(trim_img, new_shape)
       if plot : plot_trimed_img(img_name,trim_img)       
#       # creat profile for trim_img
#       trim_img_grey = cv2.cvtColor(trim_img, cv2.COLOR_BGR2GRAY)    
#       new_profile = trim_img_grey.sum(axis=0)/new_shape[1] 
#       if plot : plot_trimed_img_profile(img_name,new_profile)      
       if savepath != '': cv2.imwrite(savepath,trim_img)        
       return trim_img             
    except Exception as e:
       print(type(e).__name__," : ",e)       
       return ''
    #####################################
#########################################
      


if __name__ == '__main__':
   #thepath = 'DSC_2991.JPG' #r=g<b #need invert
   #thepath = '21411.png' # 345 347 347 r<g=b close #do not invert
   #thepath = 'IMG_9999_th.JPG' #r=g>b
   #thepath = '135849.png' #r=g<b close
   #thepath = 'nullne_30s_corp_rotateRGB.JPG' #  r=b<g
   #thepath = '1041.png' #b<r<g
   #thepath = 'IMG_20170106_194812.jpg' #r=b>g  16 15 16 inv strong zero order
   #thepath = 'IMG_9999_full.JPG'
   #thepath = 'test_cfl.png'
   #thepath = '87494_ne.png'
   #thepath = '2936_cfl.png'
   #thepath = 'IMG_2018222.jpg' #vertical
   #thepath = 'IMG_20180829_00.jpg' #mono abs. not flip
   #thepath = 'IMG_20170118_203052.jpg' 
   #thepath = '12Lyrae.JPG' 
   #thepath = 'Sun.jpg'
   #thepath = '43521.jpg'
   #thepath = 'na.jpg'   
   #thepath = 'DSC_2624.JPG' 
   #thepath = '_DSC5154.JPG'  
   #thepath = '_DSC5181.JPG'   
   #thepath = 'he_20s_corp_rotate(RGB).jpg'   
   #thepath = 'IMG_1210.JPG'  # very slope
   #thepath = '87187.jpg' 
   #thepath = '131152.jpg' 
   thepath = '62994.jpg' 
   test = img_preprocess(imgpath=thepath, savepath='',new_shape=(480,15), auto_flip=True, plot=True)
   #aa = img_preprocess(imgpath=thepath, savepath='')


        
        
        
    



   






























