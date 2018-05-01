function M = getSampleCov(M)
%% Get the sample covariance matrix
%
%  Copyright by Bac Nguyen (bac.nguyencong@ugent.be)
 
    M = (M * M') / size(M, 2);
    M = regularize_matrix(M, 0.001);
end

% Regularize the covariance matrix
function M = regularize_matrix(M, alpha)
   if abs(det(M)) < 1e-10
      M = (1 - alpha) * M + eye(size(M)) * alpha;
   end
end