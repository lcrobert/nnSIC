# -*- coding: utf-8 -*-
"""
As Title, and handling the input dataset pipeline
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt 
#from datetime import datetime

def counter_TFRecord_datasize(path):
    counts = 0
    for record in tf.python_io.tf_record_iterator(path):
        counts += 1 
    return counts 

def chk_TFRecords_file(filepath, batch_size=1, plot=False):   
    
    def parse_single_example(record):         
        img_features = tf.parse_single_example(
                       record,
                       features={'label': tf.FixedLenFeature([], tf.int64),
                                 'image': tf.FixedLenFeature([], tf.string),
                                 'height': tf.FixedLenFeature([], tf.int64),
                                 'width': tf.FixedLenFeature([], tf.int64),
                                 'filename': tf.FixedLenFeature([], tf.string),                             
                                 })          
        image = tf.decode_raw(img_features['image'], tf.uint8)
        label = tf.cast(img_features['label'], tf.uint8) 
        filename = img_features['filename'] #need .decode('utf8')
        height = img_features['height']       
        width = img_features['width']                      
        return image, label, filename, height, width  

    # Settiing TFRecord structure     
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(parse_single_example)  # Parse the record into tensors.    
    # Setting Read data in batch mode 
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    # Start to read data
    sess = tf.Session() 
    sess.run(iterator.initializer)
    images, labels, filenames, heights, widths = sess.run(iterator.get_next())
    #print (labels, filenames, heights, widths)
    sess.close()
    
    # Reshape images to orignal shape due to that is a 1-D np array 
    images = [ images[i].reshape((heights[0], widths[0], 3)) for i in range(batch_size)  ]

    # Count data points of this file
    data_counts = counter_TFRecord_datasize(filepath)

    # Plot images if plot is True 
    if plot:        
       plt.figure(filepath.replace("\\","/").split("/")[-1]) 
       for i in range(batch_size):        
           plt.subplot(batch_size,1,i+1)
           plt.imshow(images[i])
       
    return {'DataSize': data_counts,'labels':labels, 'filenames':filenames, 
            'imgsize': [set(heights),set(widths)]}  


class InputDataPipeline(object):

    def __init__(self, sess, inputdata_path, batch_size, image_shape, label_size, epoch=0):
        self.sess = sess
        self.inputdata_path = inputdata_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.label_size = label_size
        self.epoch = epoch
           
        self.init_iterator()     

    def init_iterator(self):
        with tf.name_scope("InputData_iterator"):
          with tf.name_scope("train_readTFRecords"):
            self.train_dataset, self.train_iterator = self.Read_TFRecords_file(self.inputdata_path['train'],"train",self.batch_size['train'])
          with tf.name_scope("vali_readTFRecords"):
            self.vali_dataset, self.vali_iterator = self.Read_TFRecords_file(self.inputdata_path['vali'],"vali",self.batch_size['vali'])
          with tf.name_scope("test_readTFRecords"):
            self.test_dataset, self.test_iterator = self.Read_TFRecords_file(self.inputdata_path['test'],"test",self.batch_size['test'])
        
          # Make feedable iterator
          self.inputdata_handle = tf.placeholder(tf.string, shape=[],name='inputdata_handle')
          self.iterator = tf.data.Iterator.from_string_handle(
                          self.inputdata_handle, self.train_dataset.output_types)
          self.batch_image, self.batch_label, self.batch_filename = self.iterator.get_next()
          self.batch_image = tf.reshape(self.batch_image, [-1, self.image_shape[0], self.image_shape[1], 3])
          self.batch_label = tf.one_hot(self.batch_label, self.label_size)
        
        for x in [self.train_iterator, self.vali_iterator, self.test_iterator]:
            self.sess.run(x.initializer)
            #print (self.batch_size)

    def parse_single_example(self, record):
        img_features = tf.parse_single_example(
                       record,
                       features={'label': tf.FixedLenFeature([], tf.int64),
                                 'image': tf.FixedLenFeature([], tf.string),
                                 'height': tf.FixedLenFeature([], tf.int64),
                                 'width': tf.FixedLenFeature([], tf.int64),
                                 'filename': tf.FixedLenFeature([], tf.string),                             
                                 })
        
        image = tf.decode_raw(img_features['image'], tf.uint8) #1-D
        label = tf.cast(img_features['label'], tf.uint8) 
        filename = img_features['filename'] #need .decode('utf8') 
        return image, label, filename         
    
        
    def Read_TFRecords_file(self, filename, data_type, batchsize, image_shape=0, sh_buffer=0):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parse_single_example) 
        if data_type == "train" and self.epoch >= 1:  # train, vali, test
           dataset = dataset.batch(batchsize).repeat(self.epoch)
        else:    
           dataset = dataset.batch(batchsize).repeat()           
        the_iterator = dataset.make_initializable_iterator()
        return dataset, the_iterator


    def feeder(self, selected_iterator):
        handle = self.sess.run(selected_iterator.string_handle())
        return self.sess.run([self.batch_image, self.batch_label],
                             feed_dict={self.inputdata_handle: handle}) 
      



class InputData(object):

    def __init__(self, sess, inputdata_path, batch_size, image_shape, label_size):
        self.sess = sess
        self.inputdata_path = inputdata_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.label_size = label_size
           
        self.init_iterator()     

    def init_iterator(self):

        self.the_dataset, self.the_iterator = self.Read_TFRecords_file(self.inputdata_path,self.batch_size)
   
        self.inputdata_handle = tf.placeholder(tf.string, shape=[],name='inputdata_handle')
        self.iterator = tf.data.Iterator.from_string_handle(
                          self.inputdata_handle, self.the_dataset.output_types)
        self.batch_image, self.batch_label, self.batch_filename = self.iterator.get_next()
        self.batch_image = tf.reshape(self.batch_image, [-1, self.image_shape[0], self.image_shape[1], 3])
        self.batch_label = tf.one_hot(self.batch_label, self.label_size)
        
        self.sess.run(self.the_iterator.initializer)

    def parse_single_example(self, record):
        img_features = tf.parse_single_example(
                       record,
                       features={'label': tf.FixedLenFeature([], tf.int64),
                                 'image': tf.FixedLenFeature([], tf.string),
                                 'height': tf.FixedLenFeature([], tf.int64),
                                 'width': tf.FixedLenFeature([], tf.int64),
                                 'filename': tf.FixedLenFeature([], tf.string),                             
                                 })
        
        image = tf.decode_raw(img_features['image'], tf.uint8) #1-D
        label = tf.cast(img_features['label'], tf.uint8) 
        filename = img_features['filename'] #need .decode('utf8') 
        return image, label, filename         
    
        
    def Read_TFRecords_file(self, filename, batchsize):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parse_single_example)    
        dataset = dataset.batch(batchsize).repeat()           
        the_iterator = dataset.make_initializable_iterator()
        return dataset, the_iterator

    def feeder(self, selected_iterator):
        handle = self.sess.run(selected_iterator.string_handle())
        return self.sess.run([self.batch_image, self.batch_label],
                             feed_dict={self.inputdata_handle: handle}) 


###############################################################################
if __name__ == '__main__':   
  input_main_path = os.path.join(os.getcwd(),'InputData')   
  train_path = os.path.join(input_main_path,'MyDataV2_train.tfrecords')   
  vali_path = os.path.join(input_main_path,'MyDataV2_vali.tfrecords')   
  test_path = os.path.join(input_main_path,'MyDataV2_test.tfrecords') 

  for filepath in [train_path,vali_path,test_path]:
      filename = filepath.replace("\\","/").split("/")[-1]
      datasize = counter_TFRecord_datasize(filepath)
      print (filename,':\n','size :',datasize)
      
  """    
  my_data = chk_TFRecords_file(train_path,batch_size=4,plot=True)
  for key,value in my_data.items(): 
      print (key ,':', value)
  """  









"""
image_shape = [15,480]
label_size = 3 # 0~2 of 3 class
train_data_size = 1644 #604
vali_data_size = 411 #107
test_data_size = 240 #89


# Hyper parameter 
train_batch_size = 240 #320 240 320 640
learn_rate = 0.0001#0.0001 0.00001
fc_layer_neuro_size = 1024#720 650 512 976 1024
convolution_kernel_size = 32 #5up 21 19
epoch = 1800#3000
# Dropout rate
dropout = 1.0
  
  
###########################################################

# Read TFRecords 
with tf.name_scope("TFRecords_batch_train"):
  train_dataset = read_and_decode(file_train,"train",train_batch_size,image_shape, 
                                  sh_buffer=train_data_size)
  train_iterator = train_dataset.make_initializable_iterator() 

with tf.name_scope("TFRecords_batch_vali"):
  vali_dataset = read_and_decode(file_vali,"vali",vali_data_size,image_shape, 
                                 sh_buffer=vali_data_size)
  vali_iterator = vali_dataset.make_initializable_iterator() 

with tf.name_scope("TFRecords_batch_test"):
  test_dataset = read_and_decode(file_test,"test",test_data_size,image_shape, 
                                 sh_buffer=0)
  test_iterator =test_dataset.make_initializable_iterator()  

with tf.name_scope("TFRecords_iterator"):# Make feedable iterator     
# Use the `output_types` and `output_shapes` properties of either
# `train_dataset` or `vali_dataset` , because they have identical structure. 
  handle = tf.placeholder(tf.string, shape=[])
  iterator = tf.data.Iterator.from_string_handle(
             handle, train_dataset.output_types, train_dataset.output_shapes)
  batch_image, batch_label, filename_batch = iterator.get_next()
  batch_image = tf.reshape(batch_image, [-1, image_shape[0], image_shape[1], 3])
  batch_label = tf.one_hot(batch_label, label_size)     
  


###########################################################
###########################################################
model_save_path = os.path.join(os.getcwd(),
                               'Save','Models','colab','MyDataV2_m1',
                               'MyDataV2-1st_cnn_end_v2')
model_save_path = os.path.join(os.getcwd(),
                               'Save','Models','mypc','the_best',
                               'MyDataV2-2ed_cnn_best_v3')
model_save_path = os.path.join(os.getcwd(),
                               'Save','Models','mypc','others',
                               'MyDataV2-1st_cnn_best_v7')
#keep_prob = tf.placeholder(tf.float32)

#tf.reset_default_graph()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
with tf.Session() as sess:
     sess.run(init)
     # Restore Variables     
     print('-------------------------------------------')
     print('Restore Model Variables') 
     reStartTime = datetime.now()
     # Use import_meta_graph to load meta data
     saver = tf.train.import_meta_graph(model_save_path+'.meta')
     saver.restore(sess, model_save_path)
     print ("restore done!!!!")
     graph = tf.get_default_graph()
     #for op in graph.get_operations():
     #   if 'Placeholder' in op.name: print(op.name)               
     # Get collection data in meta
     x = tf.get_collection("input")[0]# x
     y = tf.get_collection("output")[0]# y_pred
     y_propb = tf.get_collection("y_propb")[0] # y_fc
     train_step = tf.get_collection("train_steps")[0]
     #keep_prob = graph.get_tensor_by_name('Fully-Connected-Layer-2/Placeholder:0')#for 1st model
     keep_prob = graph.get_tensor_by_name('Fully-Connected-Layer-1/Placeholder:0')#for 2ed model



     reEndTime = datetime.now()     
     print ('Restore Model Time usage : %s \n'%str(reEndTime-reStartTime))
     print('-------------------------------------------')
     ###############################################################################  
     # Calculate accuracy of test-set.
     print('Calculate accuracy of test-set:') 
     StartTime = datetime.now()     
     # 取得數據前,將所有iterator全部初始化
     sess.run(test_iterator.initializer)
     #for x in [train_iterator, vali_iterator, test_iterator]:
     #     sess.run(x.initializer)
     # 取得要使用的數據. The `Iterator.string_handle()` method returns a tensor 
     #handle_train = sess.run(train_iterator.string_handle())
     #handle_val = sess.run(vali_iterator.string_handle())
     handle_test = sess.run(test_iterator.string_handle())  
     image_batch, label_batch = sess.run([batch_image, batch_label], 
                                          feed_dict={handle: handle_test})
     y_fc = sess.run(y_propb, feed_dict = {x: image_batch*(1./255), 
                                           keep_prob: 1.0})
     # accuracy
     correct_prediction = tf.equal(tf.argmax(y_fc, 1), tf.argmax(label_batch, 1))
     test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     # confusion
     y_pred = tf.argmax(y_fc, 1) # 取得模型的max 預測label
     confusion = tf.confusion_matrix(labels=tf.argmax(label_batch, 1), predictions=y_pred, num_classes=label_size)     
     print ('Test-set Acc: %4.2f%%'%(test_accuracy.eval()*100))  
     print ('Test-set Confusion Matrix: ')  
     print (confusion.eval())
     EndTime = datetime.now()     
     print ('Time usage : %s \n'%str(EndTime-StartTime))
     print('-------------------------------------------')
     ###############################################################################  
     # Using Real-world image
     print('Using Real-world image') 
          
     imgs_list = ['test_cfl.png',
                  '2936_cfl.png',
                  'Sun.jpg',
                  '12Lyrae.JPG',
                  'IMG_20170106_cfl.jpg',
                  'P_20180714_131302_cfl.jpg',
                  'P_20180714_131302.cfl_conti.jpg',
                  '646.png',
                  '2336.png',
                  '8973.jpg',
                  '_DSC5202.JPG']
     true_labels = [1,1,2,2,1,1,1,0,1,0,1]
     pred_labels = [] 
     from function_imgPreprocess_single import img_preprocess     
     for img_paths in imgs_list: 
         
         img = img_preprocess(imgpath=img_paths, savepath='', plot=False)        
         if len(img) != 0:  
            image_shape = img.shape 
            img = img.reshape((-1, image_shape[0], image_shape[1], 3))           
            #print (img.shape)
            result_label = sess.run(y, feed_dict = {x: img*(1./255), keep_prob: 1.0})
            result_propb = sess.run(y_propb, feed_dict = {x: img*(1./255), keep_prob: 1.0})
            pred_labels.append(result_label[0])
            print(img_paths+' :')
            print('  Label =', result_label)
            print('  Propblity =',['%.3f'%(p) for p in result_propb[0]])
         else:
            print(img_paths+' : ','False of img_preprocess')
     print ('True label : ', true_labels)
     print ('Pred label : ', pred_labels)    
     correct = np.equal(true_labels, pred_labels).astype(int)
     acc = np.mean(correct)
     print ('Real-world-image Test Acc: %4.2f%%'%(acc*100)) 
     print('-------------------------------------------')


     ###########################################################
     sess.close() 
     ###########################################################
     
     
     
    
###########################################################
"""



#讀取model存檔並重建模型程式

#import tensorflow as tf
#import cv2
#
#
#with tf.Session() as sess:
#
#    ##################################################
#    # load model #
#    save_path = os.path.join(os.getcwd(),'MY','demo_model')    
#
#    # 使用 import_meta_graph 載入計算圖
#    saver = tf.train.import_meta_graph(save_path+'.meta')
#
#    # 使用 restore 重建計算圖
#    saver.restore(sess, save_path)
#    
#    # 取出集合內的值
#    x = tf.get_collection("input")[0]
#    y = tf.get_collection("output")[0]
#
#    ##################################################
#    
#    # 讀一張影像
#    img = cv2.imread('test_cfl.png', 1)#0=mono
#    #2936.png  test_cfl.png
#    # 辨識影像，並印出結果
#    result = sess.run(y, feed_dict = {x: img.reshape((-1, 150*640*3))})
#    print('#################')
#
#    print(result)



















