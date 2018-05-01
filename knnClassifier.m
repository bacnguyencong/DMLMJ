function preds = knnClassifier(xTr, labels, k, xTe)
%% Perform k-nearest-neighbor classification
% INPUT:
%       xTr   : input data   (d x n)
%       labels: class labels (n x 1)
%       k     : number of nearest neighbors
%       xTe   : test data    (d x m)
% OUPUT:
%       preds :  predicted label for each instance in colum indices
%
%  Copyright by Bac Nguyen (bac.nguyencong@ugent.be)
%
%%
%     Mdl = fitcknn(xTr',labels,'NumNeighbors',k,...
%                           'IncludeTies', true, ...
%                           'DistanceWeight','inverse');
%     preds = predict(Mdl,xTe');
%                       
    ind   = kNearestNeighbors(xTr, xTe, k);    
    preds = labels(ind);    
    if size(ind, 1) > 1  %check if is 1 nearest neighbor classifier
        preds = mode(preds, 1);
    end
    preds = preds(:);
end
