function [L, params ] = DMLMJ(xTr, yTr, params)
%% Distance Metric Learning through Maximization of the getGeneralizedEigenvectors divergence
%  INPUT
%       xTr   : input data  (d x n)
%       yTr   : class label (n x 1)
%       params: 
%               .kernel (If set to 1, a kerned method is applied, default = 0)
%               .ker    (Kernel type: 'rbf' or 'poly' will be applied, default = 'rbf')
%               .knn    (Number of neighbors, default = 5)
%               .dim    (Desired number of dimensionality, default = cross-validation)
%  OUTPUT: 
%       L     :  A linear transformation
%
%  Copyright by Bac Nguyen (bac.nguyencong@ugent.be)
%
%%
    if (~exist('params', 'var'))
        params = struct();
    end

    params  = getDefaultParameters(params);
    d       = size(xTr, 1);
        
    %% with kernel trick
    % check the kernel option     
    if params.kernel ~= 0          
        if (strcmp(params.ker,'poly') )
            fprintf('Sorry, not available yet!\n');
            pause;
        else
            if (strcmp(params.ker, 'rbf') )                     
                [sigma, d]   = crossSigma(xTr, yTr, params);                
                params.sigma = sigma;                   
            end            
        end        
        X       = kernelmatrix('rbf', xTr, xTr, params.sigma);
        
        % build the difference spaces
        [S, D]  = getDifferenceSpace(X, yTr, params);    
            
    else
    %% without kernel trick            
        if isfield(params, 'dim') == 0 
            COV = cvpartition(yTr,'HoldOut',0.3);
            d = crossvalidate(xTr(:,COV.training), yTr(COV.training),...
                                 xTr(:, COV.test), yTr(COV.test), params);
        else
            d = params.dim;
        end
        % build the difference spaces
        [S, D] = getDifferenceSpace(xTr, yTr, params);
          
    end
    
    % Estimate the covariance matrices
    S = getSampleCov(S);
    D = getSampleCov(D);
    
    % Get the solution
    L = getGeneralizedEigenvectors(D, S, d);        
end


%% Auxiliar functions
%  Return classification accuracy 
function percent =  knnValidate(L, xTr, yTr, xTe, yTe, knn)
    X  = L'* xTr;
    Xt = L'* xTe;
    
    preds = knnClassifier(X, yTr, knn, Xt);
    percent = 100 * sum (preds == yTe) / length(yTe);
    
    clear('preds', 'X', 'Xt');
end

% Do cross-validation to get stable sigma
function [sigma, d] = crossSigma(xTr, yTr, params)
    fprintf('Crossvalidating sigma value for rbf kernel !!!\n');    
    k     = 5; % number of fold cross-validation
    [d, n]= size(xTr);    
    index = zeros(n, 1)==0;
    rp    = randperm(n);
    sigma = 1;
    act   = 0;    
    knn   = params.knn;
    
    dbegin= 1;
    dend  = size(xTr,1);
    
    if isfield(params, 'dim') == 1
        dbegin= params.dim;
        dend  = params.dim;        
    end
    
    for dim=dbegin:1:dend
        for g = 2.^(-15:3); % values of sigma to try        
            temp = 0;
            for i=1:k
                test_start  = ceil(n/k * (i-1)) + 1;
                test_end    = ceil(n/k * i);

                index(rp(test_start:test_end)) = false;
                % devide datasets
                xtest  = xTr(:, ~index);
                ytest  = yTr(~index);
                xtrain = xTr(:, index);
                ytrain = yTr(index);                
                % learning
                X      = kernelmatrix('rbf', xtrain, xtrain, g);
                Xt     = kernelmatrix('rbf', xtrain, xtest, g);
                [S, D] = getDifferenceSpace(X, ytrain, params);    
                S      = getSampleCov(S);
                D      = getSampleCov(D);
                
                L      = getGeneralizedEigenvectors(D, S, dim);                
                temp   = temp + knnValidate(L, X, ytrain, Xt, ytest, knn);
                
                index(rp(test_start:test_end)) = true;
            end

            temp = temp / k;        
            fprintf('Testing sigma = %.5f, obtained %.2f%c\n', g, temp, '%');                        

            if (act < temp)
                act   = temp;
                sigma = g;       
                d = dim;
            end
        end      
    end
       
    fprintf('Best sigma = %.5f, d = %d obtained  %.2f%c correct\n', sigma, d, act, '%');
end

% Do cross-validation to get stable dimension
function d = crossvalidate(xTr, yTr, xTe, yTe, params)
    
    [S, D] = getDifferenceSpace(xTr, yTr, params);
    S      = getSampleCov(S);
    D      = getSampleCov(D);
    L      = getGeneralizedEigenvectors(D, S);
    knn    = params.knn;
    
    act = 0;
    d = size(L,1);
    
    for i = size(L,1):-1:1        
        temp = knnValidate(L(:,1:i), xTr, yTr, xTe, yTe, knn);                            
        if ( temp > act )                 
            act = temp;
            d = i;
        end            
    end
    
    fprintf('Best dimension is d = %d with  %.2f%c correct\n', d, act, '%');    
end
