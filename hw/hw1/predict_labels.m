function [predicted_labels, error_rate] = predict_labels(X_train, A_test_images, B_test_labels, method_name)

% X_train are the parameters trained using the given model
% A_test_images are the images in the testing data set and B_test_labels
% are their corresponding labels.


new_labels = A_test_images * X_train;
predicted_labels = zeros(size(A_test_images,1),10);
for k = 1:size(predicted_labels, 1)
   index = find(new_labels(k,:) == max(new_labels(k,:)),1);
   predicted_labels(k, index) = 1;
%    predicted_labels(k, :) = predicted_labels(k, :) == max(predicted_labels(k, :));
end

% need the 2 in the bottom as if a label is incorrect both the true
% % position and the misclassified position will have the wrong number
% figure (1)
% subplot(121)
% error_rate = sum(sum(predicted_labels ~= B_test_labels)) / (2*size(predicted_labels, 1));
% pcolor(flipud(X_train)), shading interp;
% colormap('hot')
% ylabel('Pixel')
% xlabel('Digit')
% title(method_name)
% colorbar;
% 
% subplot(122)
% plot(reshape(X_train, [1, numel(X_train)]), 'r.', 'markersize', 5)


% tranpose is because reshape fills the columns of the new matrix first,
% but X is ordered by rows --> X = [image_row1.'; image_row2 .'; ...]
figure(2)
subplot(121)
pixel_preferences = reshape(sum(abs(X_train), 2), [28,28]).';
pcolor(flipud(pixel_preferences)), shading interp;
colormap('hot')
xlabel('x coordinate')
ylabel('y coordinate')
title(strcat("Pixel preferences for ", method_name))
colorbar;

subplot(122)
error_rate = sum(sum(predicted_labels ~= B_test_labels)) / (2*size(predicted_labels, 1));
pcolor(flipud(X_train)), shading interp;
colormap('hot')
ylabel('Pixel')
xlabel('Digit')
title(strcat("Coefficient values for ",method_name))
colorbar;
% plot(reshape(pixel_preferences, [1, numel(pixel_preferences)]), 'r.', 'markersize', 5)
% ylabel('Coefficient Value')
% xlabel('Pixel Number')


figure(3)
subplot(121)
pcolor(flipud(pixel_preferences ~= 0))
colormap(gray(2))
title('Nonzero pixel values (in white)')
xlabel('x coordinate')
ylabel('y coordinate')


subplot(122)
pcolor(flipud(X_train ~= 0)), shading interp;
colormap(gray(2))
title('Nonzero coefficient values (in white)')
ylabel('Pixel')
xlabel('Digit')


% Find the most important pixels in this image
figure(4)
num_pixels = 100; % number of pixels to extract.

% sort by value
% [sorted, I] = sort(reshape(pixel_preferences, [1, 28^2]), 'descend'); 

% sort by distance from mean
[sorted, I] = sort(abs(reshape(pixel_preferences, [1, 28^2]) - mean(mean(pixel_preferences))), 'descend');
best_pixels = zeros(1, 28^2);
best_pixels(I(1:num_pixels)) = 1;

pcolor(flipud(reshape(best_pixels, [28,28])));
colormap(gray(2));
title("Top 100 most important pixels (in white)")


end

