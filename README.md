# Melbourne Landmark Identification 
This sub-project is based on 0.75 MobileNet v1 160 (the model of inception v3 is also provided). 
To start classifictaion, simply invoke the single_image_classifier method in classifier.py.
Do not forget to specify the filename you want to recognize.
## Change Model
To change the trained model, uncomment the first line and comment the second line in classifier.py.
```python
# model_file = "./output/inception_v3_batch64_lr0_1/inception_v3_batch64_lr0.1.pb"
  model_file = "./output/mobilenet_v1_075_160_batch64_lr0_1/mobilenet_v1_batch64_lr0.1.pb"
```
## Start a Server
To start a local web server running on port 5000, simply run server_test.py. The received image will be downloaded named as uploadImg.jpg. Make sure the PC has connected to a LAN.
## Mobile Implementation - iOS
Source code and usage can be found [here](https://github.com/zhiqinzhang/MelLandmarkRecApp).
