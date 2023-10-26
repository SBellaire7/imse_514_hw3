%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% WARNING - EXPENSIVE SCRIPT TO RUN! %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;
set(0,'defaultfigurecolor',[1 1 1]);

%% Read in Data and Create Design Matrix
data = readtable("HW3-data.csv");
y = string(table2array(data(:,2)));
y(y == "M") = 1; y(y == "B") = 0; y = str2double(y);
x = table2array(data(:,3:end));
x = normalize(x, 1, 'range');
labels = string(data.Properties.VariableNames(3:end));
X = [ones(size(x,1), 1), x];

% Show Correlation Matrix for x
C = corr(x);
figure; imagesc(abs(C));
set(gca, 'XTick', 1:size(x,2), 'XTickLabel', labels, 'FontSize', 8, ...
         'YTick', 1:size(x,2), 'YTickLabel', labels);
title("Input Correlation Matrix", 'FontSize', 14, 'FontName', 'Times');

% Remove some Columns
drop = [3, 4, 13, 14, 21, 23, 24];
idx = 1:size(x,2); idx(drop) = []; labels(drop) = [];
x = x(:,idx);
X = [ones(size(x,1), 1), x];

%% Train-Test Split
trainPct = 0.80;
posIdx = find(y == 1); negIdx = find(y == 0); 
nPos = length(posIdx); nNeg = length(negIdx); 
posIdx = posIdx(randperm(nPos)); negIdx = negIdx(randperm(nNeg));
trIdx = [posIdx(1:floor(nPos*0.8)); negIdx(1:floor(nNeg*0.8))];
tstIdx = [posIdx(floor(nPos*0.8) + 1 : end); negIdx(floor(nNeg*0.8) + 1 : end)];

%% Train Logistic Regression Model Bottom-Up
learnRate = 1e-3;
maxEpochs = 500;
batchSize = 16; % Batch Size for SGD
lambda = 1e-4; % Lambda for L2 Regularizer
regularizeEpoch = 300; % Epoch to Start Regularization on
currentFeatures = [1]; % Start with Intercept and Build Up Model
remainFeatures = 2:size(X,2); % Features Remaining to Add

% Iterations to Train Each Model -> Accuracy for best model selection
% is the mean accuracy of N iterations for each model. Increasing this
% value is very computationally expensive!
numIterPerModel = 3;

% For all Remaining Features
while ~isempty(remainFeatures)

    accs = [];

    % Add 1 Remaining Feature and Train Model
    for i = 1:length(remainFeatures)

        accs(i) = 0;
        

        % Train Model 5 Times for Mean Accuracy
        for k = 1:numIterPerModel

            posIdx = posIdx(randperm(nPos)); negIdx = negIdx(randperm(nNeg));
            trIdx = [posIdx(1:floor(nPos*0.8)); negIdx(1:floor(nNeg*0.8))];
            tstIdx = [posIdx(floor(nPos*0.8) + 1 : end); negIdx(floor(nNeg*0.8) + 1 : end)];

            mdl = zeros(length(currentFeatures)+1, 1);
            dw = zeros(length(currentFeatures)+1, 1);
            f = [currentFeatures, remainFeatures(i)];
        
            Xtst = X(tstIdx, f); ytst = y(tstIdx);
            Xtr = X(trIdx,f); ytr = y(trIdx);
            ntr = size(Xtr, 1); ntst = size(Xtst, 1);
    
            for j = 1:maxEpochs
        
                % Stochasically Permute Data
                perm = randperm(ntr);
                XtrShf = Xtr(perm,:); ytrShf = ytr(perm,:);
        
                for b = 1:batchSize:ntr-batchSize
        
                    % Get Batch
                    Xbatch = XtrShf(b:b+batchSize,:);
                    ybatch = ytrShf(b:b+batchSize,:);
        
                    % Predict
                    p = 1 ./ (1 + exp(-Xbatch*mdl));
                    e = p - ybatch;
                
                    % Update Weights
                    dw(1) = sum(e);
                    dw(2:end) = Xbatch(:,2:end)'*e;
                    if j >= regularizeEpoch
                        dw = dw + lambda * mdl;
                    end
                    mdl = mdl - learnRate * dw;
        
                end
        
            end
    
            % Evaluate Performance
            p = 1 ./ (1 + exp(-Xtst*mdl));
            ypred = p > 0.5;
            accs(i) = accs(i) + mean(ytst == ypred)/numIterPerModel;

        end

    end
    
    % Select Best Feature to Add
    [maxAcc, maxIdx] = max(accs);
    currentFeatures = [currentFeatures, remainFeatures(maxIdx)];
    remainFeatures(maxIdx) = [];
    fprintf("Acc: %.3f     Features = [b,", maxAcc);
    for i = 2:length(currentFeatures) 
        if i == length(currentFeatures) fprintf("%s]\n", labels(currentFeatures(i)-1)); 
        else fprintf("%s, ", labels(currentFeatures(i)-1)); 
        end
    end

end