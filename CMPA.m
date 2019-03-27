% CMPA.m: 
%
% ELEC4700, PA-10
% Author: Jacob Godin
% Date: 2019/03/26
%--------------------------------------------------------------------------

% Task 1: Generate I Data and Plot

Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95,0.7,200);

I = Is*(exp(1.2*V/0.025) - 1) + Gp*V - Ib*exp((-1.2/0.025)*(V+Vb));
I_vari = zeros(200,1);

for i=1:length(I)
    I_low = I(i) - 0.10*I(i);
    I_high = I(i) + 0.10*I(i);
    
    I_vari(i) = (I_high-I_low)*rand() + I_low;
end


figure('Name','I-V Plot');
plot(V, I,'LineWidth',3);
hold on;
grid;
title('Current Voltage Characteristic', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);

figure('Name','I-V Plot-2');
plot(V, I_vari, 'LineWidth',1);
hold on;
grid;
title('Current Voltage Characteristic with Noise', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);

figure('Name','I-V Plot');
semilogy(V, I,'LineWidth',3);
hold on;
grid;
title('Current Voltage Characteristic', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);

figure('Name','I-V Plot-2');
semilogy(V, I_vari, 'LineWidth',1);
hold on;
grid;
title('Current Voltage Characteristic with Noise', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);

% Task 2: Polynomial Fitting

p4 = polyfit(V,I,4);
y4 = polyval(p4,V);

figure('Name','I-V Plot');
plot(V, I,'LineWidth',3);
hold on;
plot(V, y4,'LineWidth',3);
grid;
title('Current Voltage Characteristic', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);
legend('I', '4th Order Polyfit');

figure('Name','I-V Plot-2');
plot(V, I_vari, 'LineWidth',1);
hold on;
plot(V, y4, 'LineWidth',1);
grid;
title('Current Voltage Characteristic with Noise', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);
legend('I', '4th Order Polyfit');

p8 = polyfit(V,I,8);
y8 = polyval(p8,V);

figure('Name','I-V Plot');
plot(V, I,'LineWidth',3);
hold on;
plot(V, y8,'LineWidth',3);
grid;
title('Current Voltage Characteristic', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);
legend('I', '8th Order Polyfit');

figure('Name','I-V Plot-2');
plot(V, I_vari, 'LineWidth',1);
hold on;
plot(V, y8, 'LineWidth',1);
grid;
title('Current Voltage Characteristic with Noise', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);
legend('I', '8th Order Polyfit');


% Task 3: Nonlinear Curve Fitting

% Task 3a)

x = V;
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C.*(exp(1.2*(-(x+1.3))/25e-3)-1)');

ff = fit(V',I',fo);
If = ff(x);

% Task 3b)

fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C.*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(V',I',fo2);
If2 = ff(x);

% Task 3c)

fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C.*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(V',I',fo3);
If3 = ff(x);

figure('Name','I-V Plot');
plot(x, If,'LineWidth',3);
grid;
hold on;
plot(x, If2,'LineWidth',2);
hold on;
plot(x, If3,'LineWidth',1);
title('Current Voltage Characteristic, using fit()', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);
legend('A, C','A, B, C','A, B, C, D');

% Task 4: Fitting Using Neural Net Model

inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

figure('Name','I-V Plot');
plot(inputs, Inn,'LineWidth',3);
grid;
title('Current Voltage Characteristic, using Neural Net Model', 'FontSize',12);
xlabel('Voltage  (V)','FontSize',20);
ylabel('Current (mA)','FontSize',20);





