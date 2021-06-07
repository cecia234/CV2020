%% Script for T3
%Use transfer learning based on a pre-trained AlexNet
%freeze the weights of all the layers but the last fully connected layer and
%fine-tune the weights of the last layer based on the same train and validation sets employed before;
%% import dataset
LazebnikTrainDatasetPath = fullfile('train');
imds = imageDatastore(LazebnikTrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Splitting train dataset in train and validation datasets (85%-15%)
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%convert BN to 3 channel BN repeating the channel 3 times
imdsTrain.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);

imdsValidation.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
%%
net = alexnet;

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));
%%
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


%% Augment
inputSize = net.Layers(1).InputSize;
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
augimdsTrain = augmentedImageDatastore(inputSize(1:3),imdsTrain, ...
    'DataAugmentation',augmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:3),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

%%
LazebnikTestDatasetPath  = fullfile('test');

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
augimdsTest = augmentedImageDatastore(inputSize(1:3),imdsTest);
[YPred,scores] = classify(netTransfer,augimdsTest);
YValidation = imdsTest.Labels;
mean(YPred == YValidation)

%%
% confusion matrix
figure
plotconfusion(YValidation,YPred)
