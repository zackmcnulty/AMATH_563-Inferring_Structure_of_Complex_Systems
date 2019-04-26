%% AMATH 563 HW2

% Saves the hare/lynx data in the format (year past 1845)/hares/lynx
% pop_data = [0:2:58;
%             20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65;
%             32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25].';
%         
% save("input_files/pop_data.mat", 'pop_data')

%% Part 1: Sparse Regression
% Find the best ?t nonlinear, dynamical systems model to the data using 
% sparse regression.
clear all; close all; clc;
data = load('input_files/pop_data');
pop_data = data.pop_data;

% interpolate data to get larger sample size.
dt = 0.1; % time step for interpolation points
query_points = 0:dt:58; % time points for interpolation
t_vals = query_points;
x_vals = interp1(pop_data(:,1), pop_data(:,2), query_points);
y_vals = interp1(pop_data(:,1), pop_data(:,3), query_points);

% Compute numerical derivatives of data

xdot = (x_vals(3:end) - x_vals(1:end-2)) ./ (2*dt);
xdot = [(x_vals(2) - x_vals(1))/dt xdot (x_vals(end) - x_vals(end-1)) / dt];
ydot = (y_vals(3:end) - y_vals(1:end-2)) ./ (2*dt);
ydot = [(y_vals(2) - y_vals(1))/dt ydot (y_vals(end) - y_vals(end-1)) / dt];


% t = time, x = # hares, y = # lynx
all_functions = {@(t,x,y) 1
                    @(t,x,y) x
                    @(t,x,y) y
                    @(t,x,y) x.^2
                    @(t,x,y) y.^2
                    @(t,x,y) x.*y
                    @(t,x,y) x.^3
                    @(t,x,y) y.^3
                    @(t,x,y) x.^2.*y
                    @(t,x,y) x.*(y.^2)
                    @(t,x,y) sin(x)
                    @(t,x,y) sin(y)
                    };
                
function_library = zeros(length(t_vals), length(all_functions));

for i = 1:length(all_functions)
    f = all_functions{i};
    function_library(:, i) = f(t_vals, x_vals, y_vals);
end
            
%% LASSO
% function_library * X = xdot && function_library * Y = ydot

lambda = 1;
x_coeffs = lasso(function_library, xdot, 'Lambda', lambda);
y_coeffs = lasso(function_library, ydot, 'Lambda', lambda);

figure(1)
subplot(121)
bar(x_coeffs)
title('X coefficient values')

subplot(122)
bar(y_coeffs)
title('Y coefficient values')

%% Part 2: KL Divergence
% Compute the KL divergence of the best model ?t to the data.

%% Part 3: Information Criterion
% Retain three of your best ?t models and compare their AIC and BIC scores.


%% Part 4: Time Embeddings

% Time-delay embed the system and determine if there are latent variables

%% Part 5: Belousov-Zhabotinsky Chemical Oscillator
% See what you can do with the data (i.e. repeat the ?rst two steps above
clear all; close all; clc;

load('input_files/BZ.mat')

[m,n,k]=size(BZ_tensor); % x vs y vs time data
for j=1:k 
    A=BZ_tensor(:,:,j); 
    pcolor(A), shading interp, pause(0.2) 
end