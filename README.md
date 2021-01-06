# Floor recognition

## 0.How to use

<br>0.1 Train a network and save .h model using keras(2.2.4)<br>

```python3 keras_train_mnist.py```<br>

<br>0.2 Inference your model<br>

```python3 keras_inference_model_mnist.py```<br>


## 1.Why do that
<br>
SiameseNetwork can be used in face identification. I use this network to identify what floor is now during the lift working so that I can tell others such as lift robots which robots can know what floor is now and do what it want to do.
<br>

<br>
So I use SiameseNetwork to train a lift layer indentification model using mnist datasets. Then you can crop the area in your image where is the lift number and save in our file. The network can compare your cropped image and lift image which floor is similar with.
<br>
This demo can be runned about 17 fps in nano without tensorrt. You can use trt to do inference more quickly.

<br>You can look videos in YouKU as follow:<br>
```
https://v.youku.com/v_show/id_XNTA0NTIwOTEyMA==.html
```

![image](https://github.com/zhucheng725/siamesenet_keras/blob/main/result.gif)

And this is my cute nano<br>

![image](https://github.com/zhucheng725/siamesenet_keras/blob/main/nano.jpg)
