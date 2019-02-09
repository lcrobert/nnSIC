# -*- coding: utf-8 -*-
"""
Build DNN architectures
"""
import tensorflow as tf


# Define some nuron operators 
def weight_variable(shape, stddev=0.1):                
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
def bias_variable(label_size):
    return tf.Variable(tf.constant(0.001, shape=label_size))
def conv2d(x_img, w_filter):
    return tf.nn.conv2d(x_img, w_filter, strides=[1,1,1,1], padding="SAME")
def conv1d(x_img, w_filter): #tf.nn.conv1d
    return tf.nn.conv1d(x_img, w_filter, stride=1, padding="SAME")
def pooling_2x2(x): 
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
def pooling_2x2_avg(x): 
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
  


class Classifier(object):

    # Define some NN layer operators 
    
    def conv2d_layer(self, input_neuro, kernel_para=[3,3,3,5], name="Convolution2D_Layer", activate="relu"): 
        # kernel_para = [filter_x, filter_y, chennel, kernel_size]
        with tf.name_scope(name):          
          with tf.name_scope('weight'):
            self.conv_w = weight_variable(kernel_para)        
          with tf.name_scope('batch_normalization'):
            self.conv_batch_norm = tf.layers.batch_normalization(conv2d(input_neuro, self.conv_w), training=self.training, momentum=0.9)                 
          with tf.name_scope(activate):
            if activate == "relu":
               self.conv = tf.nn.relu(self.conv_batch_norm) 

    def conv1d_layer(self, input_neuro, kernel_para=[5,1,3], name="Convolution1D_Layer", activate="relu"): 
        # kernel_para = [filter_x, chennel, kernel_size]
        with tf.name_scope(name):          
          with tf.name_scope('weight'):
            self.conv_w = weight_variable(kernel_para)        
          with tf.name_scope('batch_normalization'):
            self.conv_batch_norm = tf.layers.batch_normalization(conv1d(input_neuro, self.conv_w), training=self.training, momentum=0.9)                 
          with tf.name_scope(activate):
            if activate == "relu":
               self.conv = tf.nn.relu(self.conv_batch_norm) 

    def pooling_layer(self, input_neuro, name="Pooling_Layer", avg=False, one_d=False):
        with tf.name_scope(name):
          if one_d and avg: self.pool = tf.layers.average_pooling1d(input_neuro,pool_size=2,strides=2, padding='valid') 
          if one_d and avg==False: self.pool = tf.layers.max_pooling1d(input_neuro,pool_size=2,strides=2, padding='valid') 
          if one_d==False and avg: self.pool = pooling_2x2_avg(input_neuro)  
          if one_d==False and avg==False: self.pool = pooling_2x2(input_neuro)         
                
    def flatten(self, input_neuro, name="flatten"):
        with tf.name_scope(name):
          input_shape = input_neuro.get_shape().as_list()
          #print ('shape =',input_shape) #ex. [None, 3, 240, 9]
          n_size = 1
          for i in input_shape[1:]:
              n_size *= i
          self.flatten_size = n_size  
          self.flatten = tf.reshape(input_neuro,[-1, n_size])

    def fully_connected_layer(self, input_neuro, input_neuro_size, output_neuro_size, name="FullyConnected_Layer", activate="relu"):                                    
        with tf.name_scope(name):                              
          with tf.name_scope('weight'):
            self.fc_w = weight_variable([input_neuro_size, output_neuro_size])
          with tf.name_scope('batch_normalization'):
            self.fc_batch_norm = tf.layers.batch_normalization(tf.matmul(input_neuro, self.fc_w), training=self.training, momentum=0.9)     
          with tf.name_scope(activate): 
            if activate == "relu":                                                     
               self.fc = tf.nn.relu(self.fc_batch_norm) 
               tf.summary.histogram('fc_relu_histogram', self.fc)

    def output_classification_layer(self, input_neuro, input_size, output_size, dropout_rate, name="Output_layer", activate="softmax"):             
        with tf.name_scope(name):                                     
          with tf.name_scope('dropout'):  
            self.fc_dropout = tf.nn.dropout(input_neuro, dropout_rate)    
            tf.summary.histogram('fc_dropout_histogram', self.fc_dropout)    
          with tf.name_scope('weight'):
            self.output_w = weight_variable([input_size, output_size])
          with tf.name_scope('batch_normalization'):
            self.output_batch_norm = tf.layers.batch_normalization(tf.matmul(self.fc_dropout, self.output_w), training=self.training, momentum=0.9)                     
          with tf.name_scope(activate):
            if activate == "softmax":                                        
               self.output_classifier = tf.nn.softmax(self.output_batch_norm)
               tf.summary.histogram('output_histogram', self.output_classifier) 

    def input_layer(self, name="InputData"):                                                
        # Input placeholder    
        with tf.name_scope(name):
          self.x_img = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3], name='x_img')
          self.y_label = tf.placeholder(tf.uint8, [None, self.label_size],name='y_label')                                                 

    def optimizer(self, learn_rate, true_label, pred_label, loss_name="Loss"): 
        with tf.name_scope(loss_name):
          self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=pred_label))        
        with tf.name_scope('Trainning_step'):
          self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # because using batch_normalization
          with tf.control_dependencies(self.update_ops):   
            self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
                     
    """
    def variable_summaries(var):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.histogram('histogram', var)
    """
                                                 
    def eval_accuracy(self, y_output,y_label):            
        correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def eval_confusion(self, y_output,y_label,label_size):            
        y_pred = tf.argmax(y_output, 1)
        return tf.confusion_matrix(labels=tf.argmax(y_label, 1), predictions=y_pred, num_classes=label_size)    

    
    ##########################################################################################################
                                                 
    def __init__(self, sess, **kwargs):        
        self.sess = sess
        
    def init_model(self,image_shape,label_size,convolution_kernel_size,fc_layer_neuro_size,learn_rate,tensorbord_path):
        # Get parameter
        self.image_shape = image_shape
        self.label_size = label_size
        self.convolution_kernel_size = convolution_kernel_size
        self.fc_layer_neuro_size = fc_layer_neuro_size
        self.learn_rate = learn_rate
        self.tensorbord_path = tensorbord_path
        # Build model
        self.build_architecture()
        self.build_tensorbord_summary()                                        
        self.build_saver()
        # Init all variables of graph    
        init_var = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
        self.sess.run(init_var) 
                                                 
    def hidden_layers(self): 
        # NN_layers                                                    
        self.conv2d_layer(input_neuro=self.x_img, 
                          kernel_para=[5,5,3,self.convolution_kernel_size],
                          name="Convolution2D_layer_1")
        self.pooling_layer(input_neuro=self.conv, 
                          name="Pooling_2x2_avg",avg=True)
        self.conv2d_layer(input_neuro=self.pool, 
                          kernel_para=[3,3,self.convolution_kernel_size,self.convolution_kernel_size*2],
                          name="Convolution2D_layer_2")        
        self.pooling_layer(input_neuro=self.conv, 
                          name="Pooling_2x2",avg=False)                                                   
        self.flatten(input_neuro=self.pool, 
                          name="Flatten")                                                    
        self.fully_connected_layer(input_neuro=self.flatten ,
                                   input_neuro_size=self.flatten_size,
                                   output_neuro_size=self.fc_layer_neuro_size, 
                                   name="FullyConnected_Layer_1")                                 
        self.fully_connected_layer(input_neuro=self.fc ,
                                   input_neuro_size=self.fc_layer_neuro_size,
                                   output_neuro_size=int(self.fc_layer_neuro_size/2), 
                                   name="FullyConnected_Layer_2")                                                  
                                                 
    def build_architecture(self):                                              
        # Controller placeholder of Training                                                 
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.training = tf.placeholder_with_default(False, shape=(), name='training')          
        # NN Layers 
        self.input_layer(name="InputData")      
        self.hidden_layers()                                                                                         
        self.output_classification_layer(input_neuro=self.fc, 
                                         input_size=int(self.fc_layer_neuro_size/2), 
                                         output_size=self.label_size,
                                         dropout_rate=self.keep_prob,
                                         name="OutputClassification_layer", 
                                         activate="softmax")
        # Optimizer                                                
        self.optimizer(learn_rate=self.learn_rate, 
                       true_label=self.y_label,
                       pred_label=self.output_classifier,
                       loss_name="Loss_CrossEntropy")                                          
        
    def build_tensorbord_summary(self): # ps.in CMD: tensorboard --logdir=MY                                                    
        self.accuracy = self.eval_accuracy(self.output_classifier, self.y_label)        
        self.confusion = self.eval_confusion(self.output_classifier, self.y_label, self.label_size)                                                  
                                                 
        tf.summary.scalar('train_loss', self.loss)  
        tf.summary.scalar('train_accuracy', self.accuracy)      
        self.summary_merged_train = tf.summary.merge_all()          
        self.summary_merged_vali = tf.summary.merge([tf.summary.scalar('vali_loss', self.loss),
                                                     tf.summary.scalar('vali_accuracy', self.accuracy)])        
        self.train_writer = tf.summary.FileWriter(self.tensorbord_path) 

    def build_saver(self): 
        self.saver = tf.train.Saver()
        self.saver_earlystop = tf.train.Saver()    
        tf.add_to_collection('TrainSteps', self.train_step)
        tf.add_to_collection('OutputProb', self.output_classifier)
        tf.add_to_collection('OutputLabel', tf.argmax(self.output_classifier, 1))    
        tf.add_to_collection('InputImg', self.x_img)
        tf.add_to_collection('Inputlabel', self.y_label)    
        tf.add_to_collection('Dropout', self.keep_prob)
        tf.add_to_collection('isTraining', self.training)        

########################################################################################################################
########################################################################################################################
                                                 
class ClassifierM2(Classifier):
    
    def __init__(self, sess, **kwargs):        
        self.sess = sess

    def hidden_layers(self):                                                     
        self.conv2d_layer(input_neuro=self.x_img, 
                          kernel_para=[5,5,3,self.convolution_kernel_size],
                          name="Convolution2D_layer_1")
        self.conv2d_layer(input_neuro=self.conv, 
                          kernel_para=[5,5,self.convolution_kernel_size,self.convolution_kernel_size],
                          name="Convolution2D_layer_2")        
        self.pooling_layer(input_neuro=self.conv, 
                          name="Pooling_2x2_avg",avg=True)                                                  
        self.conv2d_layer(input_neuro=self.pool, 
                          kernel_para=[3,3,self.convolution_kernel_size,self.convolution_kernel_size*2],
                          name="Convolution2D_layer_3") 
        self.pooling_layer(input_neuro=self.conv, 
                          name="Pooling_2x2",avg=False)                           
        self.flatten(input_neuro=self.pool, 
                          name="Flatten")                                                    
        self.fully_connected_layer(input_neuro=self.flatten ,
                                   input_neuro_size=self.flatten_size,
                                   output_neuro_size=self.fc_layer_neuro_size, 
                                   name="FullyConnected_Layer_1")                                 
        self.fully_connected_layer(input_neuro=self.fc ,
                                   input_neuro_size=self.fc_layer_neuro_size,
                                   output_neuro_size=int(self.fc_layer_neuro_size/2), 
                                   name="FullyConnected_Layer_2")  
      
########################################################################################################################
########################################################################################################################     
                                                 
class ClassifierM3(Classifier):
    
    def __init__(self, sess, **kwargs):        
        self.sess = sess
                                                 
    def hidden_layers(self):                                                     
        self.conv2d_layer(input_neuro=self.x_img, 
                          kernel_para=[3,3,3,self.convolution_kernel_size],
                          name="Convolution2D_layer_1")
        self.pooling_layer(input_neuro=self.conv, 
                          name="Pooling_2x2",avg=False)                                                                           
        self.flatten(input_neuro=self.pool, 
                          name="Flatten")                                                    
        self.fully_connected_layer(input_neuro=self.flatten ,
                                   input_neuro_size=self.flatten_size,
                                   output_neuro_size=self.fc_layer_neuro_size, 
                                   name="FullyConnected_Layer_1")                                 
        self.fully_connected_layer(input_neuro=self.fc ,
                                   input_neuro_size=self.fc_layer_neuro_size,
                                   output_neuro_size=int(self.fc_layer_neuro_size/2), 
                                   name="FullyConnected_Layer_2")  
                                                                                                 
########################################################################################################################
######################################################################################################################## 
  
class ClassifierM4(Classifier): #1-D CNN
    
    def __init__(self, sess, **kwargs):        
        self.sess = sess
    
    def input_layer(self, name="InputData"): 
        
        def img_profile(shape, pix_area): #[15,480,3]            
            # luma(BT.709)=R*0.2126 + G*0.7152 + B*0.0722
            # luma(BT.601)=R*0.299 + G*0.587 + B*0.114  
            ysize, xsize = shape
            #pix_gray = pix_area[:,:,:,0]*0.2126 + pix_area[:,:,:,1]*0.7152 + pix_area[:,:,:,2]*0.0722
            pix_gray = pix_area[:,:,:,0]*0.3333 + pix_area[:,:,:,1]*0.3333 + pix_area[:,:,:,2]*0.3333          
            #print (pix_gray) #(?, 15, 480)
            intensity = tf.reduce_sum(pix_gray, 1)/ysize 
            #print (intensity) #(?, 480)
            return intensity   
 
        with tf.name_scope(name):
          self.x_img = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3], name='x_img')
          
          self.x_img_profile = tf.reshape(img_profile(self.image_shape, self.x_img),[-1, self.image_shape[1] , 1], name="img2profile") # NWC formet                                         
          self.y_label = tf.placeholder(tf.uint8, [None, self.label_size],name='y_label')  
                                        
    def optimizer(self, learn_rate, true_label, pred_label, loss_name="Loss"): 
        with tf.name_scope(loss_name):
          self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=pred_label))        
        with tf.name_scope('Trainning_step'):
          self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
          with tf.control_dependencies(self.update_ops):
            learn_rate = tf.cond(tf.less(self.loss, tf.constant(1.175)),lambda: 0.000000001,lambda: learn_rate)                                                 
            self.train_step = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)                                                                                           
                                                 
    def hidden_layers(self):                                      
        self.conv1d_layer(input_neuro=self.x_img_profile, 
                          kernel_para=[5,1,self.convolution_kernel_size],
                          name="Convolution1D_layer_1")
        self.conv1d_layer(input_neuro=self.conv, 
                          kernel_para=[15,self.convolution_kernel_size,self.convolution_kernel_size*2],
                          name="Convolution1D_layer_2")
        self.conv1d_layer(input_neuro=self.conv, 
                          kernel_para=[45,self.convolution_kernel_size*2,self.convolution_kernel_size*2],
                          name="Convolution1D_layer_3")                                                 
        self.conv1d_layer(input_neuro=self.conv, 
                          kernel_para=[90,self.convolution_kernel_size*2,self.convolution_kernel_size*3],
                          name="Convolution1D_layer_4")
        self.conv1d_layer(input_neuro=self.conv, 
                          kernel_para=[105,self.convolution_kernel_size*3,self.convolution_kernel_size*3],
                          name="Convolution1D_layer_5")                                                     
        self.pooling_layer(input_neuro=self.conv, 
                           name="Pooling_2x1", avg=False, one_d=True)                                                 
        self.flatten(input_neuro=self.pool, 
                           name="Flatten")  
        self.fully_connected_layer(input_neuro=self.flatten ,
                                   input_neuro_size=self.flatten_size,
                                   output_neuro_size=self.fc_layer_neuro_size, 
                                   name="FullyConnected_Layer_1")           
        self.fully_connected_layer(input_neuro=self.fc ,
                                   input_neuro_size=self.fc_layer_neuro_size,
                                   output_neuro_size=int(self.fc_layer_neuro_size/2), 
                                   name="FullyConnected_Layer_2")  
                                                                                                  
########################################################################################################################
########################################################################################################################                                                  
                                                 

             

                                                 
########################################################################################################################
if __name__ == '__main__':
       
    my_sess = tf.Session()        
    
    classifer = ClassifierM4(my_sess) 

    classifer.init_model([15,480],
                         7,
                         32,
                         1280,
                         0.1,
                         'tensorbord_path')
    classifer.sess.close()
    pass
                                                 
    """
    
    from ReadTFRecords import InputDataPipeline as IDP 
    
    MyData = IDP(my_sess,inputdata_path,batch_size,image_shape,label_size)
                               
    image_batch, label_batch = MyData.feeder(MyData.train_iterator)
    print (label_batch.size) 
    """



















