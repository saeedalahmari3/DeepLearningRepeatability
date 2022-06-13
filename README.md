# How to Guarantee a Repeatable Result for Deep Learning Models  
This code is for the repeatability experiment conducted by Saeed Alahmari et al. from Najran University and University of South Florida. This experiment shows that repeatability of training deep learning-based segmentation model is only guaranteed when using double floating point precision. This experiment focus in the segmentation task. The paper will be linked HERE upon acceptance by the journal. 

# Getting the code from github  
Download the code or paste this command into the terminal  
`git clone https://github.com/saeedalahmari3/DeepLearningRepeatability`  
Then navigate to the code main directory  
`cd DeepLearningRepeatability`  
# Requirments  
# 1. Pytorch  
To train a deep learning model using Pytorch library, you need to create a conda environment. Here we have provided a clone of our conda environment, where exact version of the environment can be created. To create a conda environment for training Pytorch-based deep learning model (U-Net) do:  
`conda env create -f pytorch_env.yml`  
Then activate the newly created environment ...   
`conda activate saeed_pytorch2`  

# 2. Tensorflow (Keras)  
To train a deep learning model using Tensorflow (keras), you need to create a conda environment. Here we have provided a clone of our conda environment, where exact version of the environment can be created. To create a conda environment for training TF(Keras)-based deep learning model for segmentation task (U-Net) do:  
`conda env create -f kerasTF_env.yml`  
Then activate the newly created environment ...  
`conda activate TF`  

# How to run the code:  
# 1. Pytorch  
**Training**  
To train deep learning model using Pytorch for segmentation, you need to have the data stored in (.npy) format. Our data is currently available upon request. Moreover, GPU computing is required inorder to complete the training using this code.  
To run the code using **single-floating point precision** do  
`python3 ./pytorch_code/train_singlePrecision.py -d ./data -GPU 8 -m train -iter 1`  
To run the code using **double-floating point precision** do  
`python3 ./pytorch_code/train_doublePrecision.py -d ./data -GPU 8 -m train -iter 1`  

This code will be trained for 20 epochs while using 16 batch size. To run the second model do  
`python3 ./pytorch_code/train_singlePrecision.py -d ./data -GPU 8 -m train -iter 2`  
Keep doing the same for 7 iterations `-iter 7`.   
Note you can use any GPU for this experiment. If you are using SLURM for job distrubtion then have GPU to be None `-GPU None`, see run_train.sh for an example of how to run the code using SLURM. 
**Testing**  
Testing the code for each trained model can be done as follows:  
**single-floating point precision**  
`python3 ./pytorch_code/train_singlePrecision.py -d ./data -GPU 8 -m test -iter 1`  
**double-floating point precision**  
`python3 ./pytorch_code/train_doublePrecision.py -d ./data -GPU 8 -m test -iter 1`  
# 2. Tensorflow and Keras  
**Training**  
To train deep learning model using Tensorflow with Keras frontend for segmentation tasks, you need to have the data stored in a (.npy) fromat. Our data is currently available upon request. Moreover, GPU computing is required inorder to complete the training using this code.  
To run the code using **single-floating point precision** do:  
`python3 ./Keras_code/Train.py -d ./data -GPU 20 -m train -precision float32 -iter 1`  
The training will be completed for 20 epochs using batch size of 16. `-d ./data` is the path to the dataset directory `-m train` is the mode which can be *train* or *test*  
The second run (training of another model) can be done using the same command above except with changing `-iter 2`, and so on until the `-iter 7`
To run the code using **double-floating point precision** do:  
`python3 ./Keras_code/Train.py -d ./data -GPU 20 -m train -precision float64 -iter 1`  
The second run (training of another model) can be done using the same command above except with changing `-iter 2`, and so on until the `-iter 7`  
**Testing**  
**single-floating point precision**  
`python3 ./Keras_code/Train.py -d ./data -GPU 20 -m test -precision float32 -iter 1`   
**double-floating point precision**   
`python3 ./Keras_code/Train.py -d ./data -GPU 20 -m test -precision float64 -iter 1`  

