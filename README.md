# INC-BAM
Compression using Multi stage BAM for LoRa 

## python version that i use
`3.12.10`

## Clone clean Version 1.1.1
Locate in tags, and download or clone this 
```
git clone --branch v1.1.1 --depth 1 https://github.com/seanmantanhacker/INC-BAM.git
```
## Clone experimental branch (Main)
```
git clone https://github.com/seanmantanhacker/INC-BAM.git
```
## How to use
1. Every Model-experiment-? have generate, training, and main
2. run generate to generate datasets
3. run training for training model
4. run main, for testing purpose

#### - Model 01


#### - Model 02


#### - Model 03

This is the same structure as BAM and Multi BAM
support for batch training, 
And the most importantly, add torch instead of classical numpy
its increase the speed while maintain the performace
#### - Model 04
```
NOTE Bam V4
This is the same structure of BAM and Multi BAM V3
support for batch training, 
And the most importantly, add torch instead of classical numpy
its increase the speed while maintain the performace

The main difference is how to calculate error, this model diff prediction with clean signal
```