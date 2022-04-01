Instructions

This repo contains submodules. To clone this repo, run:
git clone --recursive https://github.com/IMIG-zjlab/HIDE.git

Tools Version

The supported Xilinx tools （Vitis-AI）release is 2021.2.

Contents

This reference design contains the following:
1.demo
	This folder contains run-all source code of the project.
2.dataset
	This folder contains dataset for trianning of the project.
3.host
	This folder contains files of Training DNN Model、Quantizing DNN Model and Compiling DNN Model to xmodel.
.

4.edge
	This folder contains	three folders including ScanningImage,notebook and RuningModel,which lists command lines we use on edge.
4.1ScanningImage
	This folder contains command line script to control smartcam to take photos of CT printed image.
4.2notebook
	This folder contains command line script to pre-prosess scanned CT image.
4.3RunningModel
	This folder contains command line script to deploy and run test model.
5.readme 
This file.

Documentation
For additional documentation including architecture information and build tutorials, visit:hps://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Overview
