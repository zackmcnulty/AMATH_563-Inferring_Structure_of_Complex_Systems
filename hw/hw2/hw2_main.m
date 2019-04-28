%% AMATH 563 HW2

% Saves the hare/lynx data in the format (year past 1845)/hares/lynx
% pop_data = [0:2:58;
%             20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65;
%             32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25].';
%         
% save("input_files/pop_data.mat", 'pop_data')

%% Part 1: Sparse Regression
% Find the best nonlinear, dynamical systems model to the data using 
% sparse regression.
clear all; close all; clc;
data = load('input_files/pop_data');
pop_data = data.pop_data;


% Interpolate Data for more timepoints
dt = 0.1; % interpolation time step
query_points = 0:dt:58; % time points for interpolation
t_vals = query_points;
x_vals = interp1(pop_data(:,1), pop_data(:,2), query_points);
y_vals = interp1(pop_data(:,1), pop_data(:,3), query_points);

figure(1)
plot(t_vals, x_vals, t_vals, y_vals)


% Compute numerical derivatives of data

xdot = (x_vals(3:end) - x_vals(1:end-2)) ./ (2*dt);
xdot = [(x_vals(2) - x_vals(1))/dt,    xdot,    (x_vals(end) - x_vals(end-1)) / dt];
ydot = (y_vals(3:end) - y_vals(1:end-2)) ./ (2*dt);
ydot = [(y_vals(2) - y_vals(1))/dt,   ydot,   (y_vals(end) - y_vals(end-1)) / dt];


% Define Function Library
% t = time, x = # hares, y = # lynx
function_vector = @(t,x,y) [ones(length(t), 1) x y x.^2 y.^2 x.*y x.^3 y.^3 x.^2.*y y.^2.*x x.^2.*y.^2 ...
                                 t.^2 t.^3 sin(t) cos(t) sin(x) sin(y) cos(x) cos(y) sin(x.^2) cos(x.^2) ...
                                 exp(t) exp(x) exp(y)];
function_library = function_vector(t_vals.', x_vals.', y_vals.');

% LASSO
% function_library * X = xdot && function_library * Y = ydot
% 
lambda = 0.05;
x_coeffs = lasso(function_library, xdot.', 'Lambda', lambda);
y_coeffs = lasso(function_library, ydot.', 'Lambda', lambda);

% x_coeffs = function_library \ xdot.';
% y_coeffs = function_library \ ydot.';

%
figure(2)
subplot(121)
bar(x_coeffs)
title('X coefficient values')

subplot(122)
bar(y_coeffs)
title('Y coefficient values')

% Compute predicted model based on coefficents in our sparse regression

dxdt = @(t,x,y) dot(function_vector(t,x,y), x_coeffs);
dydt = @(t,x,y) dot(function_vector(t,x,y), y_coeffs);
f = @(t,x) [ dxdt(t, x(1), x(2)); dydt(t, x(1), x(2))];

[tx, data_est] = ode45(f, t_vals, [x_vals(1); y_vals(1)]);
data_est = real(data_est);

% Plot Results for comparison

figure(3)
subplot(121)
plot(t_vals, x_vals, 'r-', tx, data_est(:,1), 'k-')
title('Hare')

legend({'True', 'Estimated'})

subplot(122)
plot(t_vals, y_vals, 'r-', tx, data_est(:,2), 'k-')
title('Lynx')
legend({'True', 'Estimated'})



%% Part 2: KL Divergence
% Compute the KL divergence of the best model to the data.
bin_size = 2;
data_range_x = min([x_vals data_est(:,1).']): bin_size : max([x_vals data_est(:,1).']);
data_range_y = min([y_vals data_est(:,2).']): bin_size : max([y_vals data_est(:,2).']);

% generate PDFs (constant offset added to avoid division by zero)
offset = 0.01;
fx = hist(x_vals, data_range_x) + offset;
fy = hist(y_vals, data_range_y) + offset;
gx = hist(data_est(:,1) , data_range_x) + offset;
gy = hist(data_est(:,2) , data_range_y) + offset;

% normalize data

fx = fx / trapz(data_range_x, fx);
fy = fy / trapz(data_range_y, fy);
gx = gx / trapz(data_range_x, gx);
gy = gy / trapz(data_range_y, gy);

% Plot distributions
figure(4)
subplot(121)
plot(data_range_x, fx, 'r-', data_range_x, gx, 'b-');
xlim([data_range_x(1), data_range_x(end) ])
legend({'Data', 'Model'})

subplot(122)
plot(data_range_y, fy, 'r-', data_range_y, gy, 'b-');
legend({'Data', 'Model'})


% Calculate KL Divergence

KL_x = trapz(fx .*log(fx ./ gx));
KL_y = trapz(fy .* log(fy ./ gy));

%% Part 3: Information Criterion
% Retain three of your best three models and compare their AIC and BIC scores.

% Estimate Log likelihood of model


%% Part 4: Time Embeddings

% Time-delay embed the system and determine if there are latent variables

%% Part 5: Belousov-Zhabotinsky Chemical Oscillator
% See what you can do with the data (i.e. repeat the first two steps above
clear all; close all; clc;

load('input_files/BZ.mat')

% [m,n,k]=size(BZ_tensor); % x vs y vs time data
% for j=1:k 
%     A=BZ_tensor(:,:,j); 
%     pcolor(A), shading interp, pause(0.2) 
% end