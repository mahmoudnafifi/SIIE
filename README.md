# Sensor Independent Illumination Estimation for DNN Models
*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1</sup> and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>2</sup>
<br></br><sup>1</sup>York University  <sup>2</sup>Samsung Research

<img src="https://drive.google.com/uc?export=view&id=1wwu-vpAl1mh8qcXqvhTpJHGlxuaam-Me" style="width: 350px; max-width: 100%; height: auto" title="Click for the larger version." />


### Abstract
While modern deep neural networks (DNNs) achieve state-of-the-art results for illuminant estimation, it is currently necessary to train a separate DNN for each type of camera sensor. This means when a camera manufacturer uses a new sensor, it is necessary to re-train an existing DNN model with training images captured by the new sensor. This paper addresses this problem by introducing a novel sensor-independent illuminant estimation framework. Our method learns a sensor-independent <i>working space</i> that can be used to canonicalize the RGB values of any arbitrary camera sensor. Our learned space retains the linear property of the original sensor raw-RGB space and allows unseen camera sensors to be used on a single DNN model trained on this working space.  We demonstrate the effectiveness of this approach on several different camera sensors and show it provides performance on par with state-of-the-art methods that were trained per sensor.

### Test our trained models
To test our trained model, run `demo.m` after changing the model `model_name` and `image_name` variables based on your seleced trained model and input image filename. You can test any of our trained models located in `models` directory. Each model was trained using different camera sensors as discussed in the paper. Each model is named based on the validation set used during the training (for example, the model `trained_model_wo_CUBE+_CanonEOS550D.mat` was trained on all images from [NUS](http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html") and [Gehler-Shi](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) datasets without including any example from Cube/Cube+ datasets. The input image's PNG file <b><i>must</i></b> be used after applying black/saturation normalization. This is very important since all trained networks expect to get normalized input images. 

### FAQ
1. #### Can I use it to correct my sRGB JPEG images?
No, our method works with linear raw-RGB images. The black level removal should be apply to any input image. To corret your sRGB images, you can check our work proposed for white balancing sRGB images from [here](http://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/index.html).

2. #### Can I images captured by different camera sensors different than the NUS, Gehler-Shi, Cube+ camera sensors?
Yes, our work propose to reduce the camera sensor bias effects that most of learning-based illuminant estimation models suffer from. Our method targets to work independently from the camera sensor. Read our paper for more details

3. #### Where can I find the training code?
Currently, the training code is not provided. 

4. #### Can it be used for different proposes than illuminant estimation?

5. #### How to integrate the RGB-*uv* histogram block to my network? 




### Publication
Mahmoud Afifi and Michael S. Brown, Sensor Independent Illumination Estimation for DNN Models, British Machine Vision Conference (BMVC), 2019.




