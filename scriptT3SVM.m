%% Script T3 
% employ the pre-trained network as a feature extractor, accessing the
% activation of an intermediate layer (for instance the last convolutional layer) and train a multiclass linear SVM. For implementing
% the multiclass SVM use any of the approaches seen in the lectures,
% for instance DAG.
%% import network
net = alexnet;

%% import dataset
LazebnikTrainDatasetPath = fullfile('train');
LazebnikTestDatasetPath  = fullfile('test');

imds = imageDatastore(LazebnikTrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Splitting train dataset in train and validation datasets (85%-15%)
quotaForEachLabel=0.85;

[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

% Rescale images
imdsTrain.ReadFcn = @(x)imresize(imread(x),[64 64]);
%convert BN to 3 channel BN repeating the channel 3 times
imdsTrain.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
imdsValidation.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);

imdsTest = imageDatastore(LazebnikTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Rescale images
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);
%convert BN to 3 channel BN repeating the channel 3 times
imdsTest.ReadFcn = @(x)repmat(imread(x), 1, 1, 3);
inputSize = net.Layers(1).InputSize;
augmenter = imageDataAugmenter( ...
    'RandXReflection',1);
augimdsTrain = augmentedImageDatastore(inputSize(1:3),imdsTrain, ...
   'DataAugmentation',augmenter);
augimdsTest = augmentedImageDatastore(inputSize(1:3),imdsTest);

%% extract activations

layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%% Using svm
X = featuresTrain;
Y = imdsTrain.Labels;

%% Select each combination
numberOfClassifiers = 105; %n=105 k=2
classes = unique(imdsTrain.Labels);
combinations = nchoosek(classes,2);
SVMModels = cell(numberOfClassifiers,1);
%% Train all the 1v1 classifiers
rng(1); % For reproducibility
tic
for i = 1:numberOfClassifiers
    class1=combinations(i,1);
    class2=combinations(i,2);    
    SVMModels{i} = fitcsvm(X,Y,'ClassNames',[class1 class2],...
        'KernelFunction','rbf','BoxConstraint',1);
end
toc


%% Extract number of Support Vector as a Measure of Generalization Capability
generalizationCap = [];
indexes = [];
for i=1:numberOfClassifiers    
    generalizationCap(i)=size(SVMModels{i}.SupportVectors,1);
    indexes(i)=i;
end
%% List every classifier
% Every Row contains the two classes evaluated and the number of support
% vectors
CombTable = table(combinations(:,1), combinations(:,2),transpose(generalizationCap),transpose(indexes),'VariableNames',["Class1","Class2","NSV","Index"]);

%% Predict Classes for every Test Observation using DAG
YPred = [""];
tic
for i=1:length(featuresTest(:,1))
    YPred(i) = DAG(SVMModels,classes,featuresTest(i,:),CombTable);
end
toc

% Compute Accuracy And Confusion Matrix
mean(transpose(categorical(YPred)) == YTest)
figure
plotconfusion(YTest,transpose(categorical(YPred)))

%% [Optional] Fit Image Classifier using Matlab CECOC implementation

mdl = fitcecoc(featuresTrain,YTrain);

%  Classify Test Images
YPred = predict(mdl,featuresTest);


% Calculate Accuracy
mean(YPred == YTest)
% confusion matrix
figure
plotconfusion(YTest,YPred)