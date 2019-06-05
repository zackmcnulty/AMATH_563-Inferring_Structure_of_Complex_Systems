% Plot the test/training dataset ranges for clarity
x = linspace(0,1,100);

figure(1)
plot(0.5*ones(1,100), linspace(0,1,100), 'k-')
hold on;
plot(linspace(0,1,100), 0.5*ones(1,100), 'k-')
plot(x, 1/2*x + 0.25, 'r--');
plot(x, -1/2*x + 0.75, 'r--');
xlim([0,1]);
ylim([0,1]);
xticks([]);
yticks([]);

figure(2)

area(