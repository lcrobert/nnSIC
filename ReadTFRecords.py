# -*- coding: utf-8 -*-
"""
As Title, and handling the input dataset pipeline
"""
import os
import tensorflow as tf
#from datetime import datetime

def counter_TFRecord_datasize(path):
    counts = 0
    for record in tf.python_io.tf_record_iterator(path):
        counts += 1 
    return counts 


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
        height = img_features['height']       
        width = img_features['width']         
        if self.image_shape != '':
           return image, label, filename  
        else:
           return image, label, filename, height, width    
            
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
      

# Single InputData set
class InputData(InputDataPipeline):

    def __init__(self, sess, inputdata_path, batch_size, image_shape, label_size):
        self.sess = sess
        self.inputdata_path = inputdata_path
        if batch_size == None: 
           self.batch_size = counter_TFRecord_datasize(inputdata_path)
        else:   
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
                  
    def Read_TFRecords_file(self, filename, batchsize):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self.parse_single_example)    
        dataset = dataset.batch(batchsize).repeat()           
        the_iterator = dataset.make_initializable_iterator()
        return dataset, the_iterator


# Chk contents in the TFRecordsFile
class ChkTFRecordsFile(InputData): 
     
    def __init__(self, filepath, batch_size=1):
        self.filepath = filepath
        self.image_shape = '' # fn:parse_single_example used because unknown w h
        self.datasize = counter_TFRecord_datasize(filepath)
        self.sess = tf.Session() 
        self.init_iterator(batch_size) 
        
    def init_iterator(self, batch_size):
        self.batch_size = batch_size 
        self.the_dataset, self.the_iterator = self.Read_TFRecords_file(self.filepath, self.batch_size)                    
        self.sess.run(self.the_iterator.initializer) ##with self.sess.as_default():
   
    def get_data(self, new_batch_size=0, show_info=False, plot=False):
        if new_batch_size > 0: self.init_iterator(new_batch_size)           
        images, labels, filenames, heights, widths = self.sess.run(self.the_iterator.get_next())
        images = [ images[i].reshape((heights[0], widths[0], 3)) for i in range(self.batch_size)  ] 
        self.data = {'img': images,
                     'label':labels, 
                     'filename':filenames, 
                     'imgsize': [set(heights),set(widths)],
                     'datasize': self.datasize}        
        if show_info:
           print (self.filepath.replace("\\","/").split("/")[-1])             
           for key,value in self.data.items(): 
               print (key ,':', value) if key != 'img' else print (key ,':', 'skip') 
           print ('----------------------------------')    
        if plot : self.plot_imgs(images)        
        return self.data

    def get_all_data(self):       
        self.data = self.get_data(new_batch_size=self.datasize)
        return self.data

    def count_label(self, labels, only_print_result=True):
        import collections
        label_counts = collections.Counter(labels)
        label_counts = sorted(label_counts.items()) 
        if only_print_result:
           for i in label_counts : print (i)
        else:    
           for i in label_counts : print (i)
           return label_counts
    
    def get_label_distribution(self):
        self.data = self.get_all_data()
        self.label_counts = self.count_label(self.data['label'],only_print_result=False)
        return self.label_counts  

    def plot_imgs(self, images): 
        img_number = len(images) 
        if img_number < 5:
           import matplotlib.pyplot as plt            
           plt.figure(self.filepath.replace("\\","/").split("/")[-1]) 
           for i in range(img_number):        
               plt.subplot(img_number,1,i+1)
               plt.imshow(images[i])            
        else:
           print ("Could not plot OVER 4 images - Please re-select images OR re-run get_data(new_batch_size=4)") 


###############################################################################
if __name__ == '__main__':   
  input_main_path = os.path.join(os.getcwd(),'InputData')   
  train_path = os.path.join(input_main_path,'MyDataV2_train.tfrecords')   
  vali_path = os.path.join(input_main_path,'MyDataV2_vali.tfrecords')   
  test_path = os.path.join(input_main_path,'MyDataV2_test.tfrecords') 

  file = ChkTFRecordsFile(test_path,batch_size=4)
  
  data_all = file.get_all_data() 
  #file.plot_imgs(data_all['img'][0:3])
  labels_all = file.get_label_distribution()  
  data = file.get_data(new_batch_size=4, show_info=True, plot=True)  
  data2 = file.get_data(show_info=True, plot=True)  




  





























