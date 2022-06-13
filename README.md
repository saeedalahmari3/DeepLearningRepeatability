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
To train deep learning model using Pytorch for segmentation, you need to have the data stored in (.npy) format. Our data is currently available upon request. Moreover, GPU computing is required inorder to complete the training using this code.  
To run the code using single-floating point precision do  
`python3 ./pytorch_code/train_singlePrecision.py -d ./data -GPU 8 -m train -iter 1`  
This code will be trained for 20 epochs while using 16 batch size. To run the second model do  
`python3 ./pytorch_code/train_singlePrecision.py -d ./data -GPU 8 -m train -iter 2`  
Keep doing the same for 7 iterations `-iter 7`.   
Note you can use any GPU for this experiment. If you are using SLURM for job distrubtion then have GPU to be None `-GPU None` see 

