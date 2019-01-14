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




























