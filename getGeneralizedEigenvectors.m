function [ B ] = getGeneralizedEigenvectors( SB, SW, d )
%% 
%  INPUT:
%       SB   : the negative covariance matrix
%       SW   : the positive covariance matrix
%        d   : the desired number of features
%  OUTPUT:   
%       B    : The d eigenvectors corresponding to the d largest values
%  of (lambda_i  +  1/lambda_i)
%
%  Copyright by Bac Nguyen (bac.nguyencong@ugent.be)

    [B,V] = eig(SB, SW);
    [~, ind] = sort(-(diag(V) + 1./diag(V)));

    B = B(:,ind);

    if exist('d', 'var') 
        B = B(:, 1:d);
    end
end