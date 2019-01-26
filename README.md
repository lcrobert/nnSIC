# **CNN Spectral Image Classification** - TensorFlow

<img align="right" width="280" src="https://lcrobert.pythonanywhere.com/static/img/Sun_600_300.jpg"/>

Deep Learning (CNN) classification of spectral images on TensorFlow.

The DEMO website :  https://lcrobert.pythonanywhere.com/pysa/recognition/

Model newest version : 201901190924

[README.md English version](https://github.com/lcrobert/nnSIC/blob/master/README.en.md)
&nbsp;&nbsp;&nbsp;&nbsp; 

------

### 目錄:

* [簡介](#簡介)
* [資料集](#資料集)
* [CNN架構](#CNN架構)
* [訓練](#訓練)
* [模型在 TestSet上的表現](#模型在TestSet上的表現)
* [模型部署](#模型部署)
* [五四三](#五四三)

Last update : 20190127<br/>
<br/>

## 簡介

在開始前先說明一件事 ，這裡所說的光譜影像並**不是高光譜影像**，而是類似於頁首的那張太陽光譜，

光源經過光柵分光後被影像感測器所拍下的可見光光譜影像。更多的光譜影像可以參考 [這裡](https://spectralworkbench.org/) 及 [這裡](https://lcrobert.pythonanywhere.com/pysa/) 。

<br/>

回歸正題，這專案的發想，其實主要源自於光譜影像的波長校正這件事，一般來說，如果光譜影像來

自於未知的儀器或是不同的拍攝環境，那就必須得對影像重新作波長校正，所謂的波長校正就是將影

像的像素位置換算成實際的波長值，然而事實上這是一件瑣碎又麻煩的事，所以就有了把這件事情全

自動化的想法，其中的關鍵點有兩個 : 

-  辨識出影像中的光譜是屬於何種光源   

-  根據光源特性從圖譜中找出已知波長的特徵峰位置

所以這專案的主要目的就在於第一點，利用CNN模型來對常見的校正光源進行光譜影像辨識，然後將

這個辨識服務部署在雲端，供 **[PySA WebApp](https://lcrobert.pythonanywhere.com/pysa/)** 使用。此外，對我來說這也是一個學習 deep learning、

CNN、TensorFlow 的好機會，以這真實的case，完整實踐從資料收集到模型部署應用的過程。

<br/>

目前訓練的模型可以有效辨識以下 **7** 種類型的光譜影像 : 

| **Label** | **Description** |
| :---: | :---------- |
|   0   | LED |
|   1   | CCFL, Fluorescent lamp, Hg lamp  ( A light source with element mercury ) |
|   2   | Sun, Sky, Star, Incandescent lamp, Halogen lamp  ( A continuum light source ) |
|   3   | Neon lamp |
|   4   | Helium lamp |
|   5   | Hydrogen lamp |
|   6   | Nitrogen lamp |

 模型使用 1-D CNN 架構，在 test set 上有 95.8% 的正確率。

<br/>

## 資料集

### 影像來源及分類

專案所使用到的光譜影像少部分來自於我個人收集，大部分則是來自 [SpectralWorkbench](https://spectralworkbench.org/) 網站。在這

裡就不得不抱怨一下，這網站上的光譜資料品質真的很糟糕，雖然看上去有 10 萬多筆，但有意義且能

用的資料卻少得可憐，在我爬過的5萬多筆資料中，能用的只有 1/10 不到，像是標籤錯誤、影像重複、

影像內容不知所云、破圖...等等的問題一大堆，所以我花了不少時間在整理這些影像資料，也採了不少

坑。這些分類整理的細節可以參考  *s1_LabelImage.py*，整理前後的資料都會放在 RawData 目錄中。

### 影像前處理

光譜影像最主要的特徵就是其沿著光譜色散方向的光強度變化，也就是說並不需要整張影像去做辨識，

只需要擷取出上面所說的特徵即可。因此，對於每一張影像都會進行以下處理 : 

1. 確保光譜色散方向是x-axis方向，不分左右(長短波位置)。
2. 選取一個矩形區域，其x-axis方向上僅涵蓋有光譜訊號的區域；在y-axis方向則只有一小部分。
3. 影像縮放到同一大小，目前是設定為 15x480 。

詳細前處理過程可以參考  *s2_ImgPreprocess.py*  , *ImgPreprocessSingle.py* ，處理後的影像會輸出到 

InputData 目錄中 並將影像路徑及對應的標籤另存成 .csv檔。

### 生成訓練集、驗證集、測試集 

1. 首先確認資料集的標籤數量分布情形，評估是否需要做標籤更換。會有此考量是因為一是方便機動

   調整需要被分類的標籤有哪些，二是可以合併類別以減輕 Imbalanced data 的問題。

   目前的資料集標籤數量分布如下：

   |     Label     |  0   |  1   |  2   |  3   |  4   |  5   |  6   |
   | :-----------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
   | **Data size** | 506  | 598  | 280  | 119  |  54  |  81  | 103  |

2. 將上述的1741筆資料隨機打亂順序，拆分成訓練集和測試集，比例為 9 : 1 。

3. 由於資料集存在 imbalanced data 問題，所以針對訓練集使用最簡單的 over-sampled 方法，將數

   量低於 400 的類別，隨機挑選複製到滿足數量為止，然後整個訓練集再次隨機打亂順序。

4. 將訓練集拆分成訓練集和驗證集，比例為 9 : 1 。

5. 將訓練集、驗證集和測試集製作成 TFRecord 檔，輸出到 InputData 目錄中。另外，為了增加影像

   資料的差異性，會在每個 set 中隨機選取一半的影像作上下翻轉。

   |               | **Training set** | **Validation set** | **Test set** |
   | ------------- | ---------------- | ------------------ | ------------ |
   | **Data size** | 2696             | 300                | 175          |


詳細製作過程可參考  *s3_ImgToTFrecord.py*  。

<br>

## CNN架構  

老實說我是個初學者，當初最早最原始的 NN 架構是從 MLP 開始，看了一些 MNIST 的範例後就想說試

試看，在一陣瞎測試後宣告失敗。之後轉戰 CNN 架構，想說用 convolution 抓光譜影像特徵或許可行，

所以就從參考 LeNet-5 開始，經過一連串的架構修改、超參數調整、重新訓練、模型評估的循環後，發

現它不是 train 不起來，只是結果一直都不夠好，在 test set 上平均僅有 84% 的正確率，雖然在找到的

文獻中的確有人用 2-D CNN 成功做到光譜影像分類，但他的架構在我這 case 上似乎沒有那麼好 work，

所以在後期我決定放棄 2-D CNN 的做法。其實那時後我也一直在思考，或許要回到 1-D 光譜 profile 身

上找答案，針對 1-D 的資料，最常看到的應該是使用 RNN 或 1-D CNN 架構。以光譜 profile 的特性來

看，它不具有時間序列上的相對關係，比較多的是空間上的結構分布，所以使用 1-D CNN 或許比較適合

。嘗試了幾回合後，終於有了好消息，test set 正確率可達 95.80% ，比較看重的 precision 加權平均也

有 95.89% 的表現， 這也就證實了這架構應該是可行的。目前使用的架構如下圖所示 : <br/>
<img width="750" src="https://lcycblog.files.wordpress.com/2019/01/1d_cnn_architecture.png"/>

<br/>

Input 進來後，首先會連續經過 5 層包含 batch normalization 及 ReLU 操作的 1-D convolution layer，

其中每層的 kernel size 及 filter 數量會逐層增加以擴大對整體空間的感知，接著經過 max pooling ，然

後進入 2 層一樣有 batch normalization 及 ReLU 的 fully connected layer ，其中在第 2 個全連接層中

使用 dropout，最後的就是使用 softmax 為分類器的輸出層。在 optimize method 方面，使用 Adam 

來優化 softmax_cross_entropy。

上述的 NN 架構以及相關的 tensorbord  logger、model saver...等等設定，都會寫在 *CnnModelClass.py* 

中，有興趣的可以參考。



<br/>

## 訓練 

### 資料輸入

基本上是參考[官方文件](https://www.tensorflow.org/guide/datasets)中的 dataset 寫法，把訓練中會用到的各個資料集讀取流程，包裝成一個統一的

輸入管線，透過指定 dataset iterator 的方式在不同的資料集間做切換。資料集內的資料在這個管線中

大致會經過以下流程 :

1. 依照batch大小，從 TFRecord 中解析出相對應數量的影像及標籤。這裡不使用 randomly shuffling

   因為資料在資料集內已是 randomly，要再使用的話意義應該不大。

2. 影像資料 reshape 成原本大小。

3. 標籤作 one-hot 編碼。

關於資料輸入流程及 TFrecord 檔案的讀取應用，都寫在 *ReadTFRecords.py* 中。

### 訓練流程

這部分還沒想好要怎麼寫比較好，就直接以程式碼來說明了。

1. 設定參數

   ```bash
   - 固定參數 : 
       資料集路徑、模型儲存路徑、影像大小、類別數量  
   - 超參數 : 
       batch_size = 200
       learn_rate = 0.1
       convol_layer1_kernel_number = 32
       fc_layer1_neuro_size = 1280
       dropout_rate = 0.5 
   - 訓練步數:
       iter_number or epoch ( 基本上是以 iter_number 為主)
   ```

2. 執行訓練 (使用訓練集和驗證集)

   ```python
   def RunTraining(iter_number,earlystop=True,evalValiConfusion=False): 
       from ReadTFRecords import InputDataPipeline as IDP                
       from CnnModelClass import ClassifierM4  
       
       sess = tf.Session()            
       data = IDP(sess,inputdata_path,batch_size,image_shape,label_size,epoch=0)    
       model = ClassifierM4(sess) 
       model.init_model(image_shape, label_size, convolution_kernel_size,
                        fc_layer_neuro_size, learn_rate, save_main_path)
                            
       # Init early_stop param (vali_acc based)
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
              summary, train_loss, train_accuracy = model.sess.run([model.summary_merged_train,
                                                                    model.loss, model.accuracy],
                                                 feed_dict={model.x_img: image_batch*(1./255), 
                                                            model.y_label: label_batch,
                                                            model.keep_prob: 1.0})
       
              image_batch, label_batch = data.feeder(data.vali_iterator) 
              summary_vali, vali_loss, vali_accuracy = model.sess.run([model.summary_merged_vali,
                                                                       model.loss, model.accuracy],
                                                 feed_dict={model.x_img: image_batch*(1./255), 
                                                            model.y_label: label_batch,
                                                            model.keep_prob: 1.0})                
              
              model.train_writer.add_summary(summary, current_iterations) 
              model.train_writer.add_summary(summary_vali, current_iterations)         
              record_idx.append(current_iterations)            
              train_acc_val.append(train_accuracy)       
              vali_acc_val.append(vali_accuracy)           
              train_loss_val.append(train_loss)       
              vali_loss_val.append(vali_loss)                 
   
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
              # Plot confusion_matrix and convert to tensorboard summary  
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
   ```

3. 跑測試集

   ```python
   def RunTestSet(model,data,save=True):     
       image_batch, label_batch = data.feeder(data.test_iterator)                
       test_accuracy, confusion_mtx = model.sess.run([model.accuracy,model.confusion],
                                                feed_dict={model.x_img: image_batch*(1./255), 
                                                           model.y_label: label_batch,
                                                           model.keep_prob: 1.0})         
       model.sess.close()      
       print ('Test-set Acc: %4.2f%%'%(test_accuracy*100))  
       print ('Test-set ConfusionMatrix: \n',confusion_mtx)
   ```

以上就是我一般用來作模型訓練的流程，其他的一些小細節，比如紀錄儲存、模型驗證，這邊就先不提。

相關的程式碼，可以參考  *s4_TrainMain.py* 、*ModelTools.py*、*CnnModelClass.py*。另外，值得一提的是，

其實我主要是在 CoLab上跑模型訓練，而因為 CoLab 的環境及 jupyter notebook 的特性，實際在 CoLab

上跑的訓練流程會稍微不一樣，這之後有空會再說明。



<br/>

## 模型在TestSet上的表現

<img width="510" src="https://lcycblog.files.wordpress.com/2019/01/confusionmatrix.png">
<img width="520" src="https://lcycblog.files.wordpress.com/2019/01/cm_ana.png"><br/>

在自動波長校正這件事情上，比較重視的是預測結果的精確性，因為校正錯誤比無法校正還嚴重，無法

校正還可以用人工方式補救，而校正錯誤卻會導至系統後續輸出的結果變成垃圾。因此在評估模型的時

後會以 precision 為首要考量。另外，標籤 1 和 3 所代表的光譜是最常被拿來做為波長校正使用，所以

這兩個類別的 f1_score 也會是考量的點。<br/>

### 同場加映 : 對於真正未知影像的預測結果

因為有 PySA WebApp 的關係，我手上有一些來自於外部 ( aka 未來的潛在使用者 ) 的光譜影像，這些影

像並未加入任何資料集中，而是拿來作為模型的內部  A/B test。

對於這樣來源的影像，目前模型有 90% 的正確率，加權平均的 precision 也有 91%，而對於 cfl 和 neon 

更是達到100%。雖然統計的樣本數很少，但這樣的結果也是可以作為評估是否上線的參考。

<img width="520" src="https://lcycblog.files.wordpress.com/2019/01/myrealimgsconfusionmatrixpb_90.00.png"/><br/>

<br/>

## 模型部署

基本上就是把模型包裝成 API 服務，需要辨識的時候 call 一下，然後 API 回傳辨識結果。目前各大雲端

平台都有提供類似這樣的模型部署服務，但是我暫時還用不到它們，一來是因為我這case規模很小，使

用情境又單純，不太需要用到那麼 powerful 的運算服務；二來是因為本來就打算要練習實作這支 API，

然後部署到已經相當孰悉的 pythonanywhere 平台上，之後再以 PySA WebApp 串接上去。

<br/>

### 製作可用於部署的 .pb file

如果只需要模型給出預測，而沒有要再度訓練模型，那就把模型內的參數凍結，只取出預測所需要的參

數 ( 通常是輸入和輸出 ) 另轉成 .pb 檔即可，這會有效的降低模型的檔案大小，同時這 .pb 檔也可用於

 Google 的雲端部署方案中。詳細的製作過程可以參考 *s5_DeployPbFile.py* 以及 *ModelTools.py* 中的 

 *RestoreModelVariable* 、 *FreezeGraphToPB*  方法。

### 製作 API

這邊我是用 Django 製作， 簡單的 demo code 可以參考 *s6_DeployWithDjango.py*  以及 *ModelTools.py* 

中的 OnlineModel 類別。在實際的運用上，目前 API 的任務很單純，基本上只需要執行身分驗證、接

收數據、模型預測、結果回覆這 4 項，模型會在 server 啟動時被載入，當 API 接收到被允許的 ajax post

請求後，開始執行預測程序，預測結束後，以 Json 回應預測結果。



<br/>

## 五四三   

> 一些關於專案的隨筆紀錄
>
> <br/>
>
> ```bash
> # 另一種評估模型的思路
> 對輸入的影像進行翻轉，
> 分別得到上下翻轉及左右翻轉的影像，
> 將這三種影像分別丟入模型作預測，
> 如果預測結果有兩個以上相同，則判定為該結果，
> 否則視為unknown
> 
> ```
>
> <br/>
>
> ```bash
> # Unknown class的問題
> 對於真正不屬於任何類別的影像，它應該被歸類為unknown。
> - 以設定機率閥值的方式，將低於閥值的預測判定為unknown
> - 訓練一個二元分類器放在資料輸入端，直接對影像作判定，
>   可用one class SVM 或 AutoEncoder
> 
> ```
>
> <br/>
>
> ```bash
> # Colab early stoping 問題 (已掛載GoogleDrive)
> 當新模型要覆蓋掉舊模型的時候，
> 系統是把舊模型丟到GoogleDrive的垃圾桶中，
> 不是直接覆寫，也不是直接刪除，
> 這會導致GoogleDrive的可用空間有機會變得不夠用，
> 尤其是當訓練參數很多，模型檔案很大，epoch也多的時候。
> 
> ```
>
> <br/>
>
> ```bash
> # Future work
> - unknown class
> - 第二重點
> - 系統整合
> - model版控
> - code refactoring
> 
> ```
>
> <br/>
>
> ```bash
> # 畫神經結構圖的好工具
> - https://github.com/yu4u/convnet-drawer
>   符合需求，設定conv結構的自由度高，改一下code就可以調整輸出內容
> - http://alexlenail.me/NN-SVG/
>   線上工具， 畫FCNN及2D-CNN首選
> - https://github.com/gwding/draw_convnet
>   風格OK，自由度稍嫌不足
> 
> ```
>
> <br/>
>
> ```bash
> # tf note
> - 當需要重載計算圖時
>   tf.reset_default_graph() 要在 tf.Session() 前使用 
>   
> ```
>
> <br/>
>
> ```bash
> 
> 
> ```
>
> 



<br/><br/><br/><br/>