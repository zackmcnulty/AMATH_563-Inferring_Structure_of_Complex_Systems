function result = load_MNIST_file(file, file_type)

% https://stackoverflow.com/questions/24127896/reading-mnist-image-database-binary-file-in-matlab

% pixels are organized row-wise (when we read them in we get the pixels in
% row 1 from left to right, pixels in row 2 from left to right, etc...)
% Thus the first 28 pixels in each column of A are the first row.


% file_type = 'image' or 'label' tells whether these files correspond to
% images or labels
% file = path to file from current working directory.

%//Open file
fid = fopen(file, 'r');

%//Read in magic number
next = fread(fid, 1, 'uint32');
magicNumber = swapbytes(uint32(next));

%//Read in total number of images or labels
next = fread(fid, 1, 'uint32');
numElements = swapbytes(uint32(next));

if file_type == 'image'
    %//Read in number of rows
    next = fread(fid, 1, 'uint32');
    numRows = swapbytes(uint32(next));

    %//Read in number of columns
    next = fread(fid, 1, 'uint32');
    numCols = swapbytes(uint32(next));
    
    % stores a column for each image
    result = zeros(28^2,numElements);
else
    % stores a column for each image label.
    % See the homework specification for the format of these labels
    result = zeros(10, numElements);
    numRows = 1;
    numCols = 1;
end

for k = 1 : numElements
    %//Read in numRows*numCols pixels at a time
    next = fread(fid, numRows*numCols, 'uint8');
    if file_type == 'image'
        
        result(:, k) = imcomplement(uint8(next)); % take image complement
        
%         imshow(uint8(reshape(result(:,k), numCols, numRows)).');
%         pause(1)
    else
        next = uint8(next);
        if next == 0
            next = 10;
        end
        result(next, k) = 1;
    end
end

%//Close the file
fclose(fid);
end

