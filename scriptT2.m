%% Script for Task 2
%Improving results applying different techniques on the network obtained
%from task 1 (inserted again below)

%% import dataset
LazebnikTrainDatasetPath = fullfile('train');
imds = imageDatastore(LazebnikTrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Splitting train dataset in train and validation datasets (85%-15%)
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

% Rescale images
imdsTrain.ReadFcn = @(x)imresize(imread(x),[64 64]);
imdsValidation.ReadFcn = @(x)imresize(imread(x),[64 64]);

LazebnikTestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

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

% 1.Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

minibatches_size = 32;
options = trainingOptions('sgdm', ...
     'InitialLearnRate',0.0005,...
    'ValidationData',imdsValidation,...
    'ValidationPatience',4,...
    'MiniBatchSize',minibatches_size, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% Evaluate on test set

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

%% 2. Add Batch Normalization Layers before ReLu Layers (50%)
rng(1)
layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')%stride 1 by default
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')%stride 1 by default
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];

% Training Phase
minibatches_size = 32;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.005,...
    'ValidationData',imdsValidation,...
    'ValidationPatience',4,...
    'MiniBatchSize',minibatches_size, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

%Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

%% 2b. Add Dropout before FC Layer (40%)
rng(1)

layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')%stride 1 by default
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')%stride 1 by default
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    
    dropoutLayer(0.4,'Name','drop1')
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];

% Training Phase
minibatches_size = 32;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.005,...
    'ValidationData',imdsValidation,...
    'ValidationPatience',4,...
    'MiniBatchSize',minibatches_size, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

%Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

%% Changing size of convolutional layers 3-5-7 52%
rng(1)
layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,32,'Padding','same','Name','conv_2')%stride 1 by default
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,64,'Padding','same','Name','conv_3')%stride 1 by default
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];

% Training Phase
%Parameters
%stochastic gradient descent with momentum
minibatches_size = 32;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.005,...
    'ValidationData',imdsValidation,...
    'ValidationPatience',4,...
    'MiniBatchSize',minibatches_size, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');
    
  

%Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)
%% Increasing number of convolutional layers 3-5-7-9-11 57.4%
rng(1) 
layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,32,'Padding','same','Name','conv_2')%stride 1 by default
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,64,'Padding','same','Name','conv_3')%stride 1 by default
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')
    
    convolution2dLayer(9,32,'Padding','same','Name','conv_4')%stride 1 by default
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_4')
    
    convolution2dLayer(11,32,'Padding','same','Name','conv_5')%stride 1 by default
    batchNormalizationLayer('Name','BN5')
    reluLayer('Name','relu_5')
    
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];

% Training Phase
%Parameters
%stochastic gradient descent with momentum
minibatches_size = 128;

options = trainingOptions('sgdm', ...
     'InitialLearnRate',0.005,...   
    'ValidationData',imdsValidation,...  
    'ValidationPatience',1,...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize',minibatches_size, ...
    'Plots','training-progress',...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',15);
      

%Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)


%% Increasing number of convolutional layers 3-5-7-9-11 +dropout 59.5%
rng(1) 
layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,32,'Padding','same','Name','conv_2')%stride 1 by default
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,64,'Padding','same','Name','conv_3')%stride 1 by default
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')
    
    convolution2dLayer(9,32,'Padding','same','Name','conv_4')%stride 1 by default
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_4')
    
    convolution2dLayer(11,32,'Padding','same','Name','conv_5')%stride 1 by default
    batchNormalizationLayer('Name','BN5')
    reluLayer('Name','relu_5')
    
    dropoutLayer('Name','drop1')
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];

% Training Phase
%Parameters
%stochastic gradient descent with momentum
minibatches_size = 128;

options = trainingOptions('sgdm', ...
     'InitialLearnRate',0.005,...   
    'ValidationData',imdsValidation,...  
    'ValidationPatience',1,...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize',minibatches_size, ...
    'Plots','training-progress',...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',15);
      

%Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

%% Adam Optimizer 59.5%

% Training Phase
%Parameters
%stochastic gradient descent with momentum
minibatches_size = 128;

options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'SquaredGradientDecayFactor',0.99, ...
    'MiniBatchSize',minibatches_size, ...
    'Plots','training-progress')
    
%%Augment data
imageSize = [64 64 1];
%[Xtrain,Ytrain] = imdsTrain;
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
imdsTrainAugmented = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',augmenter);

% Training on augmented data
net = trainNetwork(imdsTrainAugmented,layers,options);

% apply the network to the test set
YPredicted1 = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

%% Ensemble Of Networks 67%
%Employ the same network on different training datasets, given by randomly
%splitting the training set at eatch network training

numberOfNetworks = 10;

% import dataset
close all force

LazebnikTrainDatasetPath = fullfile('train');

imds = imageDatastore(LazebnikTrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrainDatasets = {};
imdsValidationDatasets = {};
% Splitting train dataset in train and validation datasets (85%-15%)
quotaForEachLabel=0.85;
for i=1:numberOfNetworks
    [imdsTrainDatasets{i},imdsValidationDatasets{i}] = splitEachLabel(imds,quotaForEachLabel,'randomize');
end

% Rescale images in Validation Datasets, Training Dataset is resized in
% Data augmenter
for i=1:numberOfNetworks
    imdsValidationDatasets{i}.ReadFcn = @(x)imresize(imread(x),[64 64]);
end

LazebnikTestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

%Define Layers
layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_1',...
    'WeightsInitializer', 'narrow-normal') %stride 1 by default
    batchNormalizationLayer('Name','BN1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,32,'Padding','same','Name','conv_2')%stride 1 by default
    batchNormalizationLayer('Name','BN2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,64,'Padding','same','Name','conv_3')%stride 1 by default
    batchNormalizationLayer('Name','BN3')
    reluLayer('Name','relu_3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')
    
    convolution2dLayer(9,32,'Padding','same','Name','conv_4')%stride 1 by default
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_4')
    
    convolution2dLayer(11,32,'Padding','same','Name','conv_5')%stride 1 by default
    batchNormalizationLayer('Name','BN5')
    reluLayer('Name','relu_5')
    
    dropoutLayer('Name','drop1')
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')
    ];
%% Training Ensemble 
  rng(1)
%%Augment data
imageSize = [64 64 1];
augmenter = imageDataAugmenter('RandXReflection',1);

trainedNetworks={};

for i=1:numberOfNetworks
    minibatches_size = 128;
    options = trainingOptions('sgdm', ...
         'InitialLearnRate',0.005,...   
        'ValidationData',imdsValidationDatasets{i},...  
        'ValidationPatience',2,...
        'ExecutionEnvironment','parallel',...
        'MiniBatchSize',minibatches_size, ...
        'Plots','training-progress',...
        'MaxEpochs',50, ...
        'Shuffle','every-epoch', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',15);
    
    trainedNetworks{i} =  trainNetwork(augmentedImageDatastore(imageSize,imdsTrainDatasets{i},'DataAugmentation',augmenter),layers,options);
end


%% Evaluate on test set
LazebnikTestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set
predictedValues={};
scores={};

for i=1:numberOfNetworks
   [predictedValues{i},scores{i}] =  classify(trainedNetworks{i},imdsTest);   
end

YTest = imdsTest.Labels;
classes = unique(YTest);
% Calculate mean scores
sumOfScores = scores{1};
for i= 2:numberOfNetworks
    sumOfScores = sumOfScores + scores{i};
end

meanOfScores = sumOfScores/15;

% Predict based on mean of network scores
YPred=[""];
for i = 1:height(meanOfScores)
    [max_val, index_of_max] = max(meanOfScores(i:i,:));
    YPred(i,1)=classes(index_of_max);    
end
YPred = categorical(YPred);

%% Display Accuracy
%single accuracies
accuracies = {}
for i=1:numberOfNetworks
   accuracies{i} = mean(predictedValues{1,i} == YTest)  
end

% overall accuracy
mean(YPred == YTest)

% confusion matrix
figure
plotconfusion(YTest,YPred)

%% Optionals
%Additional Data Augmentation

%%Augment data
imageSize = [64 64 1];

imdsTrainCropped = imdsTrain;
targetSize = [48 48];
%randomly crop images
imdsTrainCropped.ReadFcn=@(x)randomCropWindow2d(size(x),targetSize);
%append images to train datastore
new_imds = imageDatastore(cat(1, imdsTrain.Files, imdsTrainCropped.Files)); 
new_imds.Labels = cat(1, imdsTrain.Labels, imdsTrainCropped.Labels);

%add a random rotation
augmenter = imageDataAugmenter( ...
    'RandXReflection',1,...
    'RandRotation',[1 5]);
imdsTrainAugmentedCropAndRotate = augmentedImageDatastore(imageSize,new_imds,'DataAugmentation',augmenter);

%
 minibatches_size = 64;

options = trainingOptions('sgdm', ...
     'InitialLearnRate',0.005,...   
    'ValidationData',imdsValidation,...  
    'ValidationPatience',4,...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize',minibatches_size, ...
    'Plots','training-progress',...
    'MaxEpochs',50, ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',15);
net = trainNetwork(imdsTrainAugmentedCropAndRotate,layers,options);
% Evaluate on test set

LazebnikTestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
mean(YPredicted == YTest) % Accuracy = 55.3%

% confusion matrix
figure
plotconfusion(YTest,YPredicted)




