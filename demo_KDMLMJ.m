load('data/rings.mat');

fprintf('Demo of KDMLMJ on the ring data set\n');
fprintf('\n----------------------------------------------\n');

% setup the parameters
params        = struct();
params.knn    = 5; % number of neighbors
params.k1     = 5;
params.k2     = 5;
params.dim    = 2; % the desired number of features

% Learn a simple linear transformation with DMLMJ
L1 = DMLMJ(xTr, yTr, params);

% Apply the kernel trick with KDMLMJ
params.kernel = 1; % RBF kernel is used
[L2, params]  = DMLMJ(xTr, yTr, params);

X = kernelmatrix('rbf', xTr, xTr, params.sigma); 

% Plot data in the transformed spaces
% Euclidean
subplot(1,3, 1)
gscatter(xTr(1,:), xTr(2,:), yTr, 'rb','o*'), legend('off'), ...
 axis square, set(gca,'Box','on'), title('Euclidean')...

xTrKer   = L2' * X;
% KDMLMJ
subplot(1,3, 2)
gscatter(xTrKer(1,:), xTrKer(2,:), yTr, 'rb','o*'), legend('off'), ...
 axis square, set(gca,'Box','on'), title('KDMLMJ')......

xTrLin   = L1' * xTr;
% DMLMJ
subplot(1,3, 3)
gscatter(xTrLin(1,:), xTrLin(2,:), yTr, 'rb','o*'), legend('off'), ...
 axis square, set(gca,'Box','on'), title('DMLMJ')......
 