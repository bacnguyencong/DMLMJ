load('data/balance.mat');

fprintf('Demo of DMLMJ on the balance data set\n');
fprintf('\n----------------------------------------------\n');

% setup the parameters
params        = struct();
params.knn    = 5; % number of neighbors
params.k1     = 5;
params.k2     = 5;
params.kernel = 0; % no kernel trick is used

% training DMLMJ
L = DMLMJ(xTr, yTr, params);

% classification accuracy
pred1 = knnClassifier(L'*xTr, yTr, 5, L'*xTe);
pred2 = knnClassifier(xTr, yTr, 5, xTe);

fprintf('\n----------------------------------------------\n');
fprintf('Euclidean accuracy = %.2f\n', sum(pred2 == yTe)/length(yTe)*100);
fprintf('DMLMJ accuracy = %.2f\n',     sum(pred1 == yTe)/length(yTe)*100);