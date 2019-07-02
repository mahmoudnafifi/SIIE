# Sensor Independent Illumination Estimation for DNN Models
*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1</sup> and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>2</sup>
<br></br><sup>1</sup>York University  <sup>2</sup>Samsung Research
<br></br>[Project page](http://cvil.eecs.yorku.ca/projects/public_html/sensor_ind_ill_est)

<img src="https://drive.google.com/uc?export=view&id=1wwu-vpAl1mh8qcXqvhTpJHGlxuaam-Me" style="width: 350px; max-width: 100%; height: auto" title="Click for the larger version." />


### Abstract
<p align="justify">While modern deep neural networks (DNNs) achieve state-of-the-art results for illuminant estimation, it is currently necessary to train a separate DNN for each type of camera sensor. This means when a camera manufacturer uses a new sensor, it is necessary to re-train an existing DNN model with training images captured by the new sensor. This paper addresses this problem by introducing a novel sensor-independent illuminant estimation framework. Our method learns a sensor-independent <i>working space</i> that can be used to canonicalize the RGB values of any arbitrary camera sensor. Our learned space retains the linear property of the original sensor raw-RGB space and allows unseen camera sensors to be used on a single DNN model trained on this working space.  We demonstrate the effectiveness of this approach on several different camera sensors and show it provides performance on par with state-of-the-art methods that were trained per sensor.</p>

### Qualitative Results
<img src="https://drive.google.com/uc?export=view&id=1EkQ4LM0MrCeY9JQgsAC4FRCsuKM-FEYH" style="width: 350px; max-width: 100%; height: auto" title="Click for the larger version." />


### Prerequisite
[Matlab](https://www.mathworks.com/downloads/) 2018b or higher

### Quick start
Run `install_.m`, then run `demo.m` to test our trained models. You can change the `model_name` and `image_name` variables based on the seleced trained model and input image filename. You can test any of our trained models located in `models` directory. Each model was trained using different camera sensors as discussed in the paper. Each model is named based on the validation set used during the training (for example, the model `trained_model_wo_CUBE+_CanonEOS550D.mat` was trained using all raw-RGB linear images from <a href="https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html">NUS</a> and <a href="https://www2.cs.sfu.ca/~colour/data/shi_gehler/">Gehler-Shi</a> datasets without including any example from <a href="https://ipg.fer.hr/ipg/resources/color_constancy">Cube/Cube+</a> datasets). The input image file <b><i>must</i></b> contains the image raw-RGB values after applying the black/saturation level-based normalization. This is very important since all trained networks expect to get uint16 black/saturation level-based normalized input images.

### FAQ
#### Can I use it to correct sRGB-rendered JPEG images?
No, our method works with linear raw-RGB images, not rendered images in the standard RGB (sRGB) space. Also, the black level removal should be apply to any input image. To corret your sRGB images, you can check <a href="https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/index.html">When Color Constancy Goes Wrong: 
Correcting Improperly White-Balanced Images, CVPR'19</a> for white balancing sRGB-rendered images (an online demo is provided).</p>

#### Can I test images captured by different camera sensors different than the camera sensors used for training (i.e., NUS, Gehler-Shi, and Cube/Cube+ datasets)?
Yes, our work propose to reduce the differences between camera sensor responses (that is why mostly all of learning-based illuminant estimation models are sensor-dependent and cannot generalize well for new camera sensors - see <a href="https://arxiv.org/pdf/1901.03198.pdf">On Finding Gray Pixels, CVPR'19</a> for interesting experiments highlight this point as well). Our method, however, targets to work independently from the camera sensor. Read our paper for more details.

#### How to report results of the trained models using a new set of raw-RGB images?
First, be sure that all images are in the raw-RGB linear space and the black level/saturation normalization is correctly applied to all testing images. The input images should be stored as uint16 PNG files. Then, you can use any trained model for testing. You can report results of one model or the best, mean, and worst results. You can also use all trained models and use the averaged illuminant vectors for evaluation (ensemble model).

#### Why does the demo show faint colors/dim images compared to what shown in the paper?
In the given demo, we show the raw-RGB image after white-balancing and scaling it up by a constant factor to aid visualization. In the paper, we used the full camera pipeline in <a href="https://karaimer.github.io/camera-pipeline/">A Software Platform for Manipulating the Camera Imaging Pipeline, ECCV'16</a> to render the image to the sRGB space with our estimated illuminant vector.

#### Where can I find the training code?
Currently, the training code is not publicly available and will be released soon.

#### How to integrate the RGB-*uv* histogram block into my network?
Please check examples given in `RGBuvHistBlock/add_RGB_uv_hist.m`.

### Publication
Mahmoud Afifi and Michael S. Brown, Sensor Independent Illumination Estimation for DNN Models, British Machine Vision Conference (BMVC), 2019.




