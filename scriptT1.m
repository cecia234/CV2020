%% Computer Vision - Project 3 - Ceschia Eugenio IN2000099
% This project requires the implementation of an image classifier 
% based on convolutional neural networks
% --------------------------Script for Task 1--------------------------
%Train a shallow network, accuracy required around 30%
%% Import Images
LazebnikTrainDatasetPath = fullfile('train');

imds = imageDatastore(LazebnikTrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Splitting train dataset in train and validation datasets (85%-15%)
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize')

%% Rescale images 
imdsTrain.ReadFcn = @(x)imresize(imread(x),[64 64]);
imdsValidation.ReadFcn = @(x)imresize(imread(x),[64 64]);
%% Train a shallow network from scratch according to the following specifications:
% network layout shown in Table 1:
rng(1);

layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')%stride 1 by default
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')%stride 1 by default
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];

%% Training Phase
%Parameters
%stochastic gradient descent with momentum
minibatches_size = 32;

options = trainingOptions('sgdm', ...
     'InitialLearnRate',0.0005,...
    'ValidationData',imdsValidation,...
    'MiniBatchSize',minibatches_size, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress')
%% train the classifier
rng(1)
net = trainNetwork(imdsTrain,layers,options);

%% Evaluate on test set

LazebnikTestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(imread(x))/divideby;
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)
