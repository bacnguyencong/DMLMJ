function [S, D] = getDifferenceSpace(xTr, yTr, params)
%% Get the positive difference space (S) and negative difference space (D)
%  INPUT:
%       xTr   : input data  (d x n)
%       yTr   : class laebl (n x 1)
%       params:
%               .knn    (Number of neighbors, default = 5)
%  OUTPUT:  S and D
%
%  Copyright by Bac Nguyen (bac.nguyencong@ugent.be)

    k1     = params.k1;
    k2     = params.k2;
    
    [d, n] = size(xTr);
    labels = unique(yTr);
    sim    = 0; 
    dis    = 0;
    S      = zeros(d, k1 * n);
    D      = zeros(d, k2 * n);
    %% build difference spaces
    for i=1:length(labels),          
        same = xTr(:, yTr == labels(i));
        if size(same, 2) == 0,
            continue;
        end

        % similar pairs
        k = min(k1 + 1, size(same,2));
        iknn = kNearestNeighbors(same, same, k);

        for j = 1:size(same, 2),  
            S(:, sim+1:sim+k-1) = bsxfun(@minus, same(:, iknn(2:k,j)),same(:,j));
            sim = sim + k - 1;
        end
        clear('iknn');

        %disimilar pairs
        notsame = xTr(:, yTr ~= labels(i));
        if size(notsame, 2) == 0
            continue;
        end
        k    = min(k2, size(notsame, 2));
        iknn = kNearestNeighbors(notsame, same, k);   
        for j = 1:size(same,2),            
            D(:, dis+1:dis+k) = bsxfun(@minus, notsame(:, iknn(1:k, j)), same(:, j));        
            dis = dis + k;
        end
        clear('iknn', 'same', 'notsame');
    end

    S = S(:, 1:sim);
    D = D(:, 1:dis);
end

