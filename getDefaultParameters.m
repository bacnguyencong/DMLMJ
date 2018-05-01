function params  = getDefaultParameters(params)
%% GETALLPARAMETERS 


%% defaut kernel setting
if isfield(params, 'kernel') == 0
    params.kernel = 0;
end

if isfield(params, 'neigh') == 0
    params.neigh = 10;
end


if isfield(params, 'ker') == 0
    params.ker = 'rbf'; 
end

if isfield(params, 'sigma') == 0
    params.sigma = 1;% width of the RBF kernel
end

if isfield(params, 'b') == 0
    params.b = 1; % bias in the linear and polinomial kernel
end

if isfield(params, 'd') == 0
    params.d = 1; % degree in the polynomial kernel
end

if isfield(params, 'knn') == 0
    params.knn  = 5;          % number of neighbors
end

