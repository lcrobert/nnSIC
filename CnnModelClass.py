# -*- coding: utf-8 -*-
"""
Build NN architectures
"""
import tensorflow as tf


    
class ClassifierM1(object):
    
    def __init__(self, sess, **kwargs):        
        self.sess = sess
        
    def init_model(self,image_shape,label_size,convolution_kernel_size,fc_layer_neuro_size,learn_rate,tensorbord_path):

        self.image_shape = image_shape
        self.label_size = label_size
        self.convolution_kernel_size = convolution_kernel_size
        self.fc_layer_neuro_size = fc_layer_neuro_size
        self.learn_rate = learn_rate
        self.tensorbord_path = tensorbord_path
        
        self.build_architecture()
        self.build_saver()
        self.build_tensorbord_summary()
   
        init_var = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        self.sess.run(init_var)

    def build_architecture(self):

        # Define tool functions
        def weight_variable(shape,stddev=0.1):                
            return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        
        def bias_variable(label_size):
            return tf.Variable(tf.constant(0.001, shape=label_size))
        
        def conv2d(x_img,w_filter):
            return tf.nn.conv2d(x_img, w_filter, strides=[1,1,1,1], padding="SAME")
        
        def pooling_2x2(x): 
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
        def pooling_2x2_avg(x): 
            return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        
        def pooling_4x4(x):
            return tf.nn.avg_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME")
    
        def variable_summaries(var):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
            tf.summary.histogram('histogram', var)
    
        def eval_accuracy(y_output,y_label):            
            correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy
    
        def eval_confusion(y_output,y_label,label_size):            
            y_pred = tf.argmax(y_output, 1)
            return tf.confusion_matrix(labels=tf.argmax(y_label, 1), predictions=y_pred, num_classes=label_size)

    
        # Placeholder    
        with tf.name_scope('InputData'):
          self.x_img = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3], name='x_img')
          self.y_label = tf.placeholder(tf.uint8, [None, self.label_size],name='y_label')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
          
        # NN Layers            
        with tf.name_scope('Convolution-Layer-1'): # Layer1          
          with tf.name_scope('conv1_w'):
            self.Wc1 = weight_variable([5,5,3,self.convolution_kernel_size]) # 5*5*3ch filter
            #variable_summaries(Wc1)           
          with tf.name_scope('conv1_batch_normalization'):
            self.conv1_batch_norm = tf.layers.batch_normalization(conv2d(self.x_img, self.Wc1), training=self.training, momentum=0.9)                 
          with tf.name_scope('conv1_relu'):  
            self.conv1 = tf.nn.relu(self.conv1_batch_norm)
            #variable_summaries(conv1)            
          with tf.name_scope('pool1_2x2_avg'): 
            self.pool = pooling_2x2_avg(self.conv1)  
                  
        with tf.name_scope('Convolution-Layer-2'): # Layer2
          with tf.name_scope('conv2_w'):
            self.Wc2 = weight_variable([3,3,self.convolution_kernel_size,self.convolution_kernel_size*2])
            #variable_summaries(Wc2)
          with tf.name_scope('conv2_batch_normalization'):
            self.conv2_batch_norm = tf.layers.batch_normalization(conv2d(self.pool, self.Wc2), training=self.training, momentum=0.9)                 
          with tf.name_scope('conv2_relu'):  
            self.conv2 = tf.nn.relu(self.conv2_batch_norm)
            #variable_summaries(conv2)
          with tf.name_scope('pool2_2x2'): 
            self.pool = pooling_2x2(self.conv2)
            
        with tf.name_scope('Fully-Connected-Layer-1'): # Layer3 
          neuro_size = self.fc_layer_neuro_size
          pool_size = self.pool.get_shape().as_list()
          #print ('pool_shape =',pool_size) #ex. [None, 3, 240, 9]
          self.pool_flatten = tf.reshape(self.pool,[-1, pool_size[1]*pool_size[2]*pool_size[3]])
          with tf.name_scope('fc1_w'):
            self.W_fc1 = weight_variable([pool_size[1]*pool_size[2]*pool_size[3], self.fc_layer_neuro_size])
          with tf.name_scope('fc1_batch_normalization'):
            self.fc1_batch_norm = tf.layers.batch_normalization(tf.matmul(self.pool_flatten, self.W_fc1), training=self.training, momentum=0.9)     
          with tf.name_scope('fc1_relu'):     
            self.fc = tf.nn.relu(self.fc1_batch_norm) 
            tf.summary.histogram('fc1_relu-histogram', self.fc)
          
        with tf.name_scope('Fully-Connected-Layer-2'): # Layer4 
          neuro_size = int(self.fc_layer_neuro_size/2)
          with tf.name_scope('fc2_w'):
            self.W_fc2 = weight_variable([self.fc_layer_neuro_size, neuro_size])
          with tf.name_scope('fc2_batch_normalization'):
            self.fc2_batch_norm = tf.layers.batch_normalization(tf.matmul(self.fc, self.W_fc2), training=self.training, momentum=0.9)     
          with tf.name_scope('fc2_relu'):     
            self.fc = tf.nn.relu(self.fc2_batch_norm) 
            tf.summary.histogram('fc2_relu-histogram', self.fc)
         
        with tf.name_scope('Output-Classification-layer'): # Layer5           
          with tf.name_scope('fc_dropout'):  
            self.fc = tf.nn.dropout(self.fc, self.keep_prob)    
            tf.summary.histogram('fc_dropout-histogram', self.fc)    
          with tf.name_scope('output_w'):
            self.W_op = weight_variable([neuro_size, self.label_size])
          with tf.name_scope('output_batch_normalization'):
            self.output_batch_norm = tf.layers.batch_normalization(tf.matmul(self.fc, self.W_op), training=self.training, momentum=0.9)                     
          with tf.name_scope('output_softmax'):     
            self.y_fc_output = tf.nn.softmax(self.output_batch_norm)
            tf.summary.histogram('Output-histogram', self.y_fc_output) 
            
        # Optimize method      
        with tf.name_scope('Loss-function-cross_entropy'):
          self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_label, logits=self.y_fc_output))        
        with tf.name_scope('Trainning_step'):
          self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # because using batch_normalization
          with tf.control_dependencies(self.update_ops):   
            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)            
        with tf.name_scope('accuracy'):
             self.accuracy = eval_accuracy(self.y_fc_output,self.y_label)        
        with tf.name_scope('confusion'):
             self.confusion = eval_confusion(self.y_fc_output,self.y_label,self.label_size) 

        
    def build_tensorbord_summary(self):        
        # ps.in CMD: tensorboard --logdir=MY
        tf.summary.scalar('train-loss', self.loss)  
        tf.summary.scalar('train-accuracy', self.accuracy)      
        self.summary_merged_train = tf.summary.merge_all()          
        self.summary_merged_vali = tf.summary.merge([tf.summary.scalar('vali-loss', self.loss),
                                                     tf.summary.scalar('vali_accuracy', self.accuracy)])        
        self.train_writer = tf.summary.FileWriter(self.tensorbord_path) 


    def build_saver(self): 
        self.saver = tf.train.Saver()
        self.saver_earlystop = tf.train.Saver()    
        tf.add_to_collection('TrainSteps', self.train_step)
        tf.add_to_collection('OutputPropb', self.y_fc_output)
        tf.add_to_collection('OutputLabel', tf.argmax(self.y_fc_output, 1))    
        tf.add_to_collection('InputImg', self.x_img)
        tf.add_to_collection('Inputlabel', self.y_label)    
        tf.add_to_collection('Dropout', self.keep_prob)
        tf.add_to_collection('isTraining', self.training)        


class ClassifierM2(object):
    
    def __init__(self, sess, **kwargs):        
        self.sess = sess
        
    def init_model(self,image_shape,label_size,convolution_kernel_size,fc_layer_neuro_size,learn_rate,tensorbord_path):

        self.image_shape = image_shape
        self.label_size = label_size
        self.convolution_kernel_size = convolution_kernel_size
        self.fc_layer_neuro_size = fc_layer_neuro_size
        self.learn_rate = learn_rate
        self.tensorbord_path = tensorbord_path
        
        self.build_architecture()
        self.build_saver()
        self.build_tensorbord_summary()
   
        init_var = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        self.sess.run(init_var)

    def build_architecture(self):

        # Define tool functions
        def weight_variable(shape,stddev=0.1):                
            return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        
        def bias_variable(label_size):
            return tf.Variable(tf.constant(0.001, shape=label_size))
        
        def conv2d(x_img,w_filter):
            return tf.nn.conv2d(x_img, w_filter, strides=[1,1,1,1], padding="SAME")
        
        def pooling_2x2(x): 
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
        def pooling_2x2_avg(x): 
            return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        
        def pooling_4x4(x):
            return tf.nn.avg_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME")
    
        def variable_summaries(var):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
            tf.summary.histogram('histogram', var)
    
        def eval_accuracy(y_output,y_label):            
            correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy
    
        def eval_confusion(y_output,y_label,label_size):            
            y_pred = tf.argmax(y_output, 1)
            return tf.confusion_matrix(labels=tf.argmax(y_label, 1), predictions=y_pred, num_classes=label_size)

    
        # Placeholder    
        with tf.name_scope('InputData'):
          self.x_img = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3], name='x_img')
          self.y_label = tf.placeholder(tf.uint8, [None, self.label_size],name='y_label')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
          
        # NN Layers            
        with tf.name_scope('Convolution-Layer-1'): # Layer1          
          with tf.name_scope('conv1_w'):
            self.Wc1 = weight_variable([5,5,3,self.convolution_kernel_size]) # 5*5*3ch filter
            #variable_summaries(Wc1)           
          with tf.name_scope('conv1_batch_normalization'):
            self.conv1_batch_norm = tf.layers.batch_normalization(conv2d(self.x_img, self.Wc1), training=self.training, momentum=0.9)                 
          with tf.name_scope('conv1_relu'):  
            self.conv1 = tf.nn.relu(self.conv1_batch_norm)
            #variable_summaries(conv1)            

        with tf.name_scope('Convolution-Layer-2'): # Layer2
          with tf.name_scope('conv2_w'):
            self.Wc2 = weight_variable([5,5,self.convolution_kernel_size,self.convolution_kernel_size])
            #variable_summaries(Wc2)
          with tf.name_scope('conv2_batch_normalization'):
            self.conv2_batch_norm = tf.layers.batch_normalization(conv2d(self.conv1, self.Wc2), training=self.training, momentum=0.9)                 
          with tf.name_scope('conv2_relu'):  
            self.conv2 = tf.nn.relu(self.conv2_batch_norm)
            #variable_summaries(conv2)
          with tf.name_scope('pool2_2x2_avg'): 
            self.pool = pooling_2x2_avg(self.conv2)

                  
        with tf.name_scope('Convolution-Layer-3'): # Layer3
          with tf.name_scope('conv3_w'):
            self.Wc3 = weight_variable([3,3,self.convolution_kernel_size,self.convolution_kernel_size*2])
            #variable_summaries(self.Wc3)
          with tf.name_scope('conv3_batch_normalization'):
            self.conv3_batch_norm = tf.layers.batch_normalization(conv2d(self.pool, self.Wc3), training=self.training, momentum=0.9)                 
          with tf.name_scope('conv3_relu'):  
            self.conv3 = tf.nn.relu(self.conv3_batch_norm)
            #variable_summaries(self.conv3)
          with tf.name_scope('pool3_2x2_max'): 
            self.pool = pooling_2x2(self.conv3)
            
        with tf.name_scope('Fully-Connected-Layer-1'): # Layer4
          neuro_size = self.fc_layer_neuro_size
          pool_size = self.pool.get_shape().as_list()
          #print ('pool_shape =',pool_size) #ex. [None, 3, 240, 9]
          self.pool_flatten = tf.reshape(self.pool,[-1, pool_size[1]*pool_size[2]*pool_size[3]])
          with tf.name_scope('fc1_w'):
            self.W_fc1 = weight_variable([pool_size[1]*pool_size[2]*pool_size[3], self.fc_layer_neuro_size])
          with tf.name_scope('fc1_batch_normalization'):
            self.fc1_batch_norm = tf.layers.batch_normalization(tf.matmul(self.pool_flatten, self.W_fc1), training=self.training, momentum=0.9)     
          with tf.name_scope('fc1_relu'):     
            self.fc = tf.nn.relu(self.fc1_batch_norm) 
            tf.summary.histogram('fc1_relu-histogram', self.fc)
          
        with tf.name_scope('Fully-Connected-Layer-2'): # Layer5 
          neuro_size = int(self.fc_layer_neuro_size/2)
          with tf.name_scope('fc2_w'):
            self.W_fc2 = weight_variable([self.fc_layer_neuro_size, neuro_size])
          with tf.name_scope('fc2_batch_normalization'):
            self.fc2_batch_norm = tf.layers.batch_normalization(tf.matmul(self.fc, self.W_fc2), training=self.training, momentum=0.9)     
          with tf.name_scope('fc2_relu'):     
            self.fc = tf.nn.relu(self.fc2_batch_norm) 
            tf.summary.histogram('fc2_relu-histogram', self.fc)
         
        with tf.name_scope('Output-Classification-layer'): # Layer6           
          with tf.name_scope('fc_dropout'):  
            self.fc = tf.nn.dropout(self.fc, self.keep_prob)    
            tf.summary.histogram('fc_dropout-histogram', self.fc)    
          with tf.name_scope('output_w'):
            self.W_op = weight_variable([neuro_size, self.label_size])
          with tf.name_scope('output_batch_normalization'):
            self.output_batch_norm = tf.layers.batch_normalization(tf.matmul(self.fc, self.W_op), training=self.training, momentum=0.9)                     
          with tf.name_scope('output_softmax'):     
            self.y_fc_output = tf.nn.softmax(self.output_batch_norm)#need name
            tf.summary.histogram('Output-histogram', self.y_fc_output) 
            
        # Optimize method      
        with tf.name_scope('Loss-function-cross_entropy'):
          self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_label, logits=self.y_fc_output))        
        with tf.name_scope('Trainning_step'):
          self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # because using batch_normalization
          with tf.control_dependencies(self.update_ops):   
            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)            
        with tf.name_scope('accuracy'):
             self.accuracy = eval_accuracy(self.y_fc_output,self.y_label)        
        with tf.name_scope('confusion'):
             self.confusion = eval_confusion(self.y_fc_output,self.y_label,self.label_size) 

        
    def build_tensorbord_summary(self):        
        # ps.in CMD: tensorboard --logdir=MY
        tf.summary.scalar('train-loss', self.loss)  
        tf.summary.scalar('train-accuracy', self.accuracy)      
        self.summary_merged_train = tf.summary.merge_all()          
        self.summary_merged_vali = tf.summary.merge([tf.summary.scalar('vali-loss', self.loss),
                                                     tf.summary.scalar('vali_accuracy', self.accuracy)])        
        self.train_writer = tf.summary.FileWriter(self.tensorbord_path) 


    def build_saver(self): 
        self.saver = tf.train.Saver()
        self.saver_earlystop = tf.train.Saver()    
        tf.add_to_collection('TrainSteps', self.train_step)
        tf.add_to_collection('OutputPropb', self.y_fc_output)
        tf.add_to_collection('OutputLabel', tf.argmax(self.y_fc_output, 1))    
        tf.add_to_collection('InputImg', self.x_img)
        tf.add_to_collection('Inputlabel', self.y_label)    
        tf.add_to_collection('Dropout', self.keep_prob)
        tf.add_to_collection('isTraining', self.training) 
################################################################################
if __name__ == '__main__':
    pass
    """
    my_sess = tf.Session()        
    
    
    classifer = ClassifierM1(my_sess) 
    classifer.init_model([15,480],
                         5,
                         32,
                         128,
                         0.5,
                         'test')
    
    
    from ReadTFRecords import InputDataPipeline as IDP 
    
    MyData = IDP(my_sess,inputdata_path,batch_size,image_shape,label_size)
                               
    image_batch, label_batch = MyData.feeder(MyData.train_iterator)
    print (label_batch.size) 
    """



















