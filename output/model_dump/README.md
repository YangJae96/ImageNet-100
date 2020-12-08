# ImageNet100
# 2020-2st 전자공학종합설계 Project 

## 1. Contributor
- 양재원

## 2. Version
- python3.7
- torch1.7 and torchvision 0.6
- Cuda >=10.2 
- Ubuntu 18.04
- Multiple GPU: Nvidia Titan X

## 3. Train
- Must to do: You need to unzip the ImageNet100.zip into the project imageNet-100 dir. Train and test should be unziped. 
- command: python train.py (Check config.py for datapath and hyper-parameter settings)

## 4. Test
- Command: python test.py (Check config.py for datapath and hyper-parameter settings)
- output: model accuracy

## 5. Best Model Test
- Unzip the Best_model.zip and move the models to the output/result dir
- Check config.py
- Run python test.py
