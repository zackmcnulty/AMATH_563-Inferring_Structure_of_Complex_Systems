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
title('Historical Hare and Lynx Populations')
xlabel('Time (years past 1845)')
ylabel('Population')
legend({'Hare', 'Lynx'})
set(gca, 'fontsize', 15)


% Compute numerical derivatives of data

xdot = (x_vals(3:end) - x_vals(1:end-2)) ./ (2*dt);
ydot = (y_vals(3:end) - y_vals(1:end-2)) ./ (2*dt);
x_vals = x_vals(2:end-1);
y_vals = y_vals(2:end-1);
t_vals = t_vals(2:end-1);

% Define Function Library
% t = time, x = # hares, y = # lynx
function_vector = @(t,x,y) [ones(length(t), 1) x y x.^2 y.^2 x.*y x.^3 y.^3 x.^2.*y y.^2.*x x.^2.*y.^2 ...
                                 t t.^2 t.^3 sin(t) cos(t) sin(x) sin(y) cos(x) cos(y) sin(x.^2) cos(x.^2) ...
                                 exp(t) exp(x) exp(y) ...
                                 t.^4, t.^5, t.^6 x.*0.5 exp(t).*sin(t) y*0.5];
function_library = function_vector(t_vals.', x_vals.', y_vals.');

% LASSO
% function_library * X = xdot && function_library * Y = ydot
% 
lambda = 0.045;
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
plot(t_vals, x_vals, 'r-', tx, data_est(:,1), 'k--', 'linewidth', 2)
title('Historical Hare Populations')
xlabel('Years Past 1845')
ylabel('Population')
legend({'True Hare', 'Estimated Hare'})
set(gca, 'fontsize', 15)

subplot(122)
plot(t_vals, y_vals, 'b-', tx, data_est(:,2), 'k--', 'linewidth', 2)
title('Historical Lynx Populations')
xlabel('Years Past 1845')
ylabel('Population')
legend({'True', 'Estimated'})
set(gca, 'fontsize', 15)


%%

threshold = 0.01;
Ix = find(x_coeffs > threshold);
x_coeffs(Ix)
Iy = find(y_coeffs > threshold);
y_coeffs(Iy)



%% Part 2: KL Divergence


% % Compute the KL divergence of the best model to the data.
% bin_size = 2;
% data_range_x = min([x_vals data_est(:,1).']): bin_size : max([x_vals data_est(:,1).']);
% data_range_y = min([y_vals data_est(:,2).']): bin_size : max([y_vals data_est(:,2).']);
% 
% % generate PDFs (constant offset added to avoid division by zero)
% offset = 0.01;
% fx = hist(x_vals, data_range_x) + offset;
% fy = hist(y_vals, data_range_y) + offset;
% gx = hist(data_est(:,1) , data_range_x) + offset;
% gy = hist(data_est(:,2) , data_range_y) + offset;
% 
% % normalize data
% 
% fx = fx / trapz(data_range_x, fx);
% fy = fy / trapz(data_range_y, fy);
% gx = gx / trapz(data_range_x, gx);
% gy = gy / trapz(data_range_y, gy);
% 
% % Plot distributions
% figure(4)
% subplot(121)
% plot(data_range_x, fx, 'r-', data_range_x, gx, 'b-');
% xlim([data_range_x(1), data_range_x(end) ])
% legend({'Data', 'Model'})
% 
% subplot(122)
% plot(data_range_y, fy, 'r-', data_range_y, gy, 'b-');
% legend({'Data', 'Model'})
% 
% 
% % Calculate KL Divergence
% 
% KL_x = trapz(fx .*log(fx ./ gx));
% KL_y = trapz(fy .* log(fy ./ gy));

clc; close all;

num_bins = [15, 15];

data_range_x = linspace(min([x_vals data_est(:,1).']), max([x_vals data_est(:,1).']), num_bins(1) + 1);
data_range_y = linspace(min([y_vals data_est(:,2).']), max([y_vals data_est(:,2).']), num_bins(1) + 1);
EDGES = {};
EDGES{1} = data_range_x;
EDGES{2} = data_range_y;

offset = 0.01;
true_f = hist3([x_vals.', y_vals.'], 'Edges', EDGES) + offset;
modeled_f = hist3(data_est, 'Edges', EDGES) + offset; %0.01 added to avoid division by zero

% normalize data
true_f = true_f ./ trapz(trapz(true_f));
modeled_f = modeled_f ./ trapz(trapz(modeled_f));

[X,Y] = meshgrid(data_range_x, data_range_y);

figure(4)
subplot(121)
surf(X, Y, true_f);
title('True model (data) probability distribution')
xlim([data_range_x(1), data_range_x(end)])
xlabel('Hare Population')
ylabel('Lynx Population')
zlabel('Probability')
ylim([data_range_y(1), data_range_y(end)])


subplot(122)
surf(X,Y, modeled_f);
title('Generated model probability distribution')
xlim([data_range_x(1), data_range_x(end)])
ylim([data_range_y(1), data_range_y(end)])
xlabel('Hare Population')
ylabel('Lynx Population')
zlabel('Probability')

KL_divergence = trapz(trapz((true_f .* log(true_f ./ modeled_f))));


%% Part 3: Information Criterion
% Retain three of your best three models and compare their AIC and BIC scores.

% Estimate Log likelihood of model

data = [x_vals.', y_vals.'];

n = length(x_vals);

RSS = 0;
for j = 1:length(data)
    RSS = RSS + norm(data(j, :) - data_est(j, :));
end

K = sum(x_coeffs ~= 0) + sum(y_coeffs ~= 0);
variance = RSS / n;
logL = -n/2*log(2*pi) - n/2*log(variance) - 1/(2*variance) * RSS;

AIC = 2*K -2*logL;
BIC = log(n) * K - 2*logL;


%% Part 4: Time Embeddings
close all; clc;

% Time-delay embed the system and determine if there are latent variables
tskip = 1; % number time points to stagger each measurement by
num_layers = 50; % number of staggered measurements to generate
layer_length = length(x_vals) - tskip*num_layers;
H_x = zeros(num_layers, layer_length);
H_y = zeros(num_layers, layer_length);

for j = 1:num_layers
    H_x(j, :) = x_vals(1+(j-1)*tskip: 1+(j-1)*tskip + layer_length-1);
    H_y(j, :) = y_vals(1+(j-1)*tskip: 1+(j-1)*tskip + layer_length-1);
end

[Ux, Sx, Vx] = svd(H_x, 'econ');
[Uy, Sy, Vy] = svd(H_y, 'econ');

% Plot singular values to look for latent variables

% three dominant singular values: outside variable besides just hare/lynx
% populations that plays a role? Or just time I guess
singular_x = diag(Sx) ./ max(diag(Sx));
singular_y = diag(Sy) ./ max(diag(Sy));

figure(6)
subplot(121)
plot(singular_x, 'r.', 'markersize', 15)
title("Time-Embedded Signular Values - Hare Population")
xlabel('Index j')
ylabel('Normalized Singular Value \sigma_j')

subplot(122)
plot(singular_y, 'r.', 'markersize', 15)
title("Time-Embedded Signular Values - Lynx Population")
xlabel('Index j')
ylabel('Normalized Singular Value \sigma_j')

%% Part 5: Belousov-Zhabotinsky Chemical Oscillator
% See what you can do with the data (i.e. repeat the first two steps above)
clear all; close all; clc;

load('input_files/BZ.mat')

% downsample to make working with system more computationally managable.
% take a single row of pixels?
BZ_tensor = BZ_tensor(100:250, 150:300, :);

% % This is a PDE system? Create 3D function library?
% 
% [m,n,k]=size(BZ_tensor); % x vs y vs time data
% for j=1:k 
%     A=BZ_tensor(:,:,j); 
%     pcolor(A), shading interp, pause(100) 
% end

% u(t, x, y)
% We want d/dt u = u_t --> calculate u_t one row at a time for each fixed y
 % time derivatives for each point (x,y) in the system with the fixed y
 
 % construct derivative matrices
 
dx = 1;
D=zeros(m,m); D2=zeros(m,m);
for j=1:m-1
  D(j,j+1)=1;
  D(j+1,j)=-1;
%
  D2(j,j+1)=1;
  D2(j+1,j)=1;
  D2(j,j)=-2;
end
D(m,1)=1;
D(1,m)=-1;
D=(1/(2*dx))*D;
%
D2(m,m)=-2;
D2(m,1)=1;
D2(1,m)=1;
D2=D2/(dx^2);
 
 
dt = 1; %?
LAMBDA = 0.002;

for y = [1,2]
   Xdot = zeros(m, k-2);
   X = BZ_tensor(:, y, :); % convert into a function of u(x,t)
   for jj = 1:m
       for j = 2:k-1
            Xdot(jj, j-1) = (X(jj, j+1) - X(jj, j-1)) / (2*dt);
       end
   end
   
    u=reshape( X(:,2:end-1).',(k-2)*m ,1 );

    for jj=2:k-1
       ux(jj-1,:)=((D*X(:,jj)).');  % u_x
       uxx(jj-1,:)=((D2*X(:,jj)).');  % u_xx
       u2x(jj-1,:)=((D* (X(:,jj).^2) ).');  % (u^2)_x
    end

    Ux=reshape(ux,(k-2)*m,1);
    Uxx=reshape(uxx,(k-2)*m,1);
    U2x=reshape(u2x,(k-2)*m,1);
    
    A=[u u.^2 u.^3 Ux Uxx U2x Ux.*u Ux.*Ux Ux.*Uxx];

    Udot=reshape((Xdot.'),(k-2)*m,1);

    xi=lasso(A,Udot,'Lambda',LAMBDA);
end
function_vector = [];
