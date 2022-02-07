# Sensor-Independent Illumination Estimation for DNN Models
*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1</sup> and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1,2</sup>
<br></br><sup>1</sup>York University &nbsp;&nbsp; <sup>2</sup>Samsung AI Center (SAIC) - Toronto
<br></br>[Project page](http://cvil.eecs.yorku.ca/projects/public_html/siie/index.html)

![BMVC_main](https://user-images.githubusercontent.com/37669469/76104974-411d4780-5fa2-11ea-93a9-b91e9da930a0.jpg)


### Prerequisite
1. [Matlab](https://www.mathworks.com/downloads/) 2018b or higher (recommended)
2. [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) for Matlab 2018b or higher



The original experiments were done using Matlab 2018b. In some higher Matlab versions, models designed in Matlab 2018b do not work. For this reason, we provide another version of models that should work but may give higher errors than the Matlab 2018b models. 

### UPDATE
**Strongly recommended to first try out the models for Matlab 2018b, if a compatibility error is faced, then try out the 'Matlab 2019b or higher' option.**



### Quick start

#### Matlab [![View Sensor-Independent Illuminant Estimation Using Deep Learning on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/72829-sensor-independent-illuminant-estimation-using-deep-learning)
Run `install_.m`, then run `demo.m` to test our trained models. In `demo.m`, you should select the version of Matlab by changing the value of `Matlab_ver`. The supported versions are: Matlab 2018b, Matlab 2019a, or higher. It is highly recommended to set `Matlab_ver` to Matlab 2018b (even if you use higher version). If you got compatability error, then try out with `Matlab_ver = 'Matlab 2019b'`.


You can change the `model_name` and `image_name` variables to choose between our trained models and to change input image filename, respectively. You can test any of our trained models located in `models` directory. Each model was trained using different camera sensors, as discussed in our [paper](http://cvil.eecs.yorku.ca/projects/public_html/siie/files/0105.pdf). Each model is named based on the validation set used during the training (for example, the model `trained_model_wo_CUBE+_CanonEOS550D.mat` was trained using all raw-RGB linear images from <a href="https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html">NUS</a> and <a href="https://www.cs.sfu.ca/~colour/data/shi_gehler/">Gehler-Shi</a> datasets without including any example from the `CanonEOS550D` camera in <a href="https://ipg.fer.hr/ipg/resources/color_constancy">Cube/Cube+</a> datasets). 



The input image file <b><i>must</i></b> contain the image raw-RGB values after applying the black/saturation level normalization. This is very important since all trained networks expect to get <b><i>uint16</i></b> input images after applying the black/saturation level normalization.




### FAQ
#### Can I use it to correct sRGB-rendered JPEG images?
No. Our method works with linear raw-RGB images, not camera-rendered images. To corret your sRGB-rendered images, you can check <a href="https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/index.html">When Color Constancy Goes Wrong: 
Correcting Improperly White-Balanced Images, CVPR'19</a> for white balancing sRGB-rendered images (an online demo is provided).</p>

#### Can I test images captured by camera sensors different than the camera sensors used for training (i.e., NUS, Gehler-Shi, and Cube/Cube+ datasets)?
Yes. Our work is proposed to reduce the differences between camera sensor responses (that is why mostly all of learning-based illuminant estimation models are sensor-dependent and cannot generalize well for new camera sensors -- see <a href="https://arxiv.org/pdf/1901.03198.pdf">On Finding Gray Pixels, CVPR'19</a> for interesting experiments that highlight this point). Our method, however, targets to work independently from the camera sensor. Read our [paper](http://cvil.eecs.yorku.ca/projects/public_html/siie/files/0105.pdf) for more details.

#### How to report results of the trained models using a new set of raw-RGB images?
First, be sure that all testing images are in the raw-RGB linear space and the black level/saturation normalization is correctly applied. The input images should be stored as uint16 PNG files after the normalization. Then, you can use any of our trained models for testing. You can report results of one model or the best, mean, and worst results obtained by all models. An example code to evaluate testing image set is provided in `evaluate_images.m` for Matlab. You can also use all trained models and report the averaged illuminant vectors for evaluation (i.e., ensemble model).

#### Why does the demo show faint colors compared to what is shown in the paper?
In the given demo, we show raw-RGB images after white balancing and scaling it up by a constant factor to aid visualization. In the [paper](http://cvil.eecs.yorku.ca/projects/public_html/siie/files/0105.pdf), we used the full camera pipeline in <a href="https://karaimer.github.io/camera-pipeline/">A Software Platform for Manipulating the Camera Imaging Pipeline, ECCV'16</a> to render images to the sRGB space with our estimated illuminant vector.


#### How to integrate the RGB-*uv* histogram block into my network?
For *Matlab 2018b and 2019a*, please check examples given in `RGBuvHistBlock/add_RGB_uv_hist.m`. If you will use the RGB-*uv* histogram block for sRGB-rendered images (e.g., JPEG images), you may need to tune the initalization of the scale and fall-off parameters for better results with sRGB images, as the current intalization was used for linear raw-RGB images. To tune these parameters, you can change the initalization of the scale parameter `C` in `scaleLayer.m` (line 39) and the fall-off factor `sigma` in `ExponentialKernelLayer.m` (line 43). The files `scaleLayer.m` and `ExponentialKernelLayer.m` are located in `RGBuvHistBlock` directory. For debugging, please use the `predict` function in `histOutLayer.m`. For *Matlab 2019b or higher (recommended)*, please check the `RGBuvHistBlock.m` code located in the `RGBuvHistBlock` directory to tune the scale/fall-off parameters.



### [Project page](http://cvil.eecs.yorku.ca/projects/public_html/siie/index.html)


### Publication

If you use this code, please cite our paper:


Mahmoud Afifi and Michael S. Brown, Sensor Independent Illumination Estimation for DNN Models, British Machine Vision Conference (BMVC), 2019.


```
@inproceedings{afifi2019SIIE,
  title={Sensor-Independent Illumination Estimation for DNN Models},
  author={Afifi, Mahmoud and Brown, Michael S},
  booktitle={British Machine Vision Conference (BMVC)},
  pages={},
  year={2019}
}
```

### Commercial Use
This software is provided for research purposes only. A license must be obtained for any commercial application.
