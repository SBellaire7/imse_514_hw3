clear all; clc; close all;
set(0,'defaultfigurecolor',[1 1 1]);

%% Read in Data and Create Design Matrix
data = readtable("HW3-data.csv");
y = string(table2array(data(:,2)));
y(y == "M") = 1; y(y == "B") = 0; y = str2double(y);
x = table2array(data(:,3:end));
x = normalize(x, 1, 'range');
labels = string(data.Properties.VariableNames(3:end));

% Get Correlation Matrix for x
C = corr(x);
figure; imagesc(abs(C));
set(gca, 'XTick', 1:size(x,2), 'XTickLabel', labels, 'FontSize', 8, ...
         'YTick', 1:size(x,2), 'YTickLabel', labels);
title("Input Correlation Matrix", 'FontSize', 14, 'FontName', 'Times');

% Drop Features
include = [28 1 2 25 17]; % From Bottom-Up Script
X = [ones(size(x,1), 1), x(:,include)];
fprintf("Features Selected (From Bottom-Up):\n");
for i = 1:length(include) fprintf("     %s\n", labels(include(i))); end
fprintf("\n");

%% Create Cross-Validation Fold Indices
k = 5; folds = [];
posIdx = find(y == 1); negIdx = find(y == 0); 
nPos = length(posIdx); nNeg = length(negIdx); 
pPart = floor(nPos/k); nPart = floor(nNeg/k);
posIdx = posIdx(randperm(nPos)); 
negIdx = negIdx(randperm(nNeg));
for i = 1:k
    idx = [posIdx((i-1)*pPart + 1 : i*pPart); ...
           negIdx((i-1)*nPart + 1 : i*nPart)];
    idx = idx(randperm(length(idx)));
    folds(:,i) = idx;
end

%% Train Logistic Regression Model on Each Fold
learnRate = 1e-3;
maxEpochs = 500;
batchSize = 16; % Batch Size for SGD
lambda = 1e-4; % Lambda for L2 Regularizer
regularizeEpoch = 300; % Epoch to Start Regularization on
dw = zeros(size(X,2), 1);
for i = 1:k

    mdl = zeros(size(X,2), 1);
    fprintf("Training on Fold %d/%d... ", i, k);

    j = 1:k; j = j(j ~= i);
    Xtst = X(folds(:,i), :); ytst = y(folds(:,i), :);
    Xtr = X(folds(:,j),:); ytr = y(folds(:,j), :);
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
    acc = mean(ytst == ypred);
    fprintf("Acc: %.3f\n", acc);

end
fprintf("\n");

%% Use Last Fold for Final Evaluation

% Predict
z = Xtst*mdl;
p = 1 ./ (1 + exp(-z));
ypred = p > 0.5;
acc = mean(ypred == ytst);

% Plot Transfer Characteristic (S-Curve)
figure; 
xsig = -10:0.05:10; ysig = 1 ./ (1 + exp(-xsig));
plot(xsig, ysig, 'Color', [0 0 1 0.3], 'LineWidth', 2); hold on; grid on;
scatter(z, p, 20, [1 0 0], 'filled'); xlim([-10 10]); ylim([-0.1 1.1]);
title("Classifier Transfer Characteristic", 'FontName', 'Times', 'FontSize', 12);
xlabel("z", 'FontName', 'Times', 'FontSize', 12); ylabel("p", 'FontName', 'Times', 'FontSize', 12);
clear z; hold off;

% Compute Confusion Matrix
TP = sum(ytst == 1 & ypred == 1);
TN = sum(ytst == 0 & ypred == 0);
FP = sum(ytst == 0 & ypred == 1);
FN = sum(ytst == 1 & ypred == 0);
nP = sum(ytst == 1); nN = sum(ytst == 0);

% Print Accuracy and CM
fprintf("Accuracy: %.2f\n", acc*100);

fprintf("Confusion Matrix:         FNR       = %.4f\n", FN / (FN + TP));
fprintf("                          FPR       = %.4f\n", FP / (FP + TN));
fprintf("        Pred +   Pred -   Precision = %.4f\n", TP / (TP + FP));
fprintf("True +  %6d   %6d   Recall    = %.4f\n", TP, FN, TP / (TP + FN));
fprintf("True -  %6d   %6d   F1 Score  = %.4f\n\n", FP, TN, 2*TP / (2*TP + FP + FN));

% Compute ROC Curve and AUC
ROCx = zeros(101, 1); ROCy = zeros(101, 1);
count = 1;
for j = 0.00:0.01:1.00

    % Predict
    p = 1 ./ (1 + exp(-Xtst*mdl));
    ypred = p > j;
    
    % Calculate Confusion Matrix Values
    TP = sum((ypred == 1) & (ytst == 1));
    FP = sum((ypred == 1) & (ytst == 0));
    FN = sum((ypred == 0) & (ytst == 1));
    TN = sum((ypred == 0) & (ytst == 0));
    
    % Compute ROC
    ROCx(count) = FP / (FP + TN); ROCy(count) = TP / (TP + FN);
    count = count + 1;

end

% Compute AUC
AUC = 0.5*sum(abs(ROCx(2:end) - ROCx(1:end-1)) .* (ROCy(2:end) + ROCy(1:end-1)));
fprintf("AUC: %.4f\n\n", AUC);

% Plot ROC
figure;
plot(ROCx, ROCy, 'Color', 'b', 'LineWidth', 1.5); grid on; hold on;
title("ROC Curve", 'FontSize', 14, 'FontName', 'Times');
xlabel("FPR (1 - Specificity)", 'FontName', 'Times', 'FontSize', 12);
ylabel("TPR (Sensitivity)", 'FontName', 'Times', 'FontSize', 12);
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlim([-0.05 1.05]); ylim([-0.05 1.05]);