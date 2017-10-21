clear all;
clc;
close all;

input = [];
output = [];
test_input = [];
test_output = [];

mg_data = importdata('mg30.dat');
%Divide os dados em 70pct treinamento 30 pct teste


mg_data_training = mg_data(1:1050);
mg_data_test = mg_data(1051:1500);
W = sqrt(0.01).*randn(1,size(mg_data_training,2)); %Gaussian white noise W
mg_data_training = mg_data_training + W; %Add the noise
mg_data_test = mg_data_test +W;

%Normalizando as entradas
% [mg_data_training_n] = mapminmax(mg_data_training');
% mg_data_training = mg_data_training_n';
% [mg_data_test_n] = mapminmax(mg_data_test');
% mg_data_test = mg_data_test_n';

%Dados de entrada (t-3)(t-2)(t-1)(t) Saída (t+1)
%Gerando dados


% j=1;
% for t=4:1049
%     
%     input(j,1) = mg_data_training(t-3);
%     input(j,2) = mg_data_training(t-2);
%     input(j,3) = mg_data_training(t-1);
%     input(j,4) = mg_data_training(t);
%     output(j) = mg_data_training(t+1); 
%     
%     j = j +1;
% end
% %Teste
% 
% j = 1;
% for t = 4:449
%     test_input(j,1) = mg_data_test(t-3);
%     test_input(j,2) = mg_data_test(t-2);
%     test_input(j,3) = mg_data_test(t-1);
%     test_input(j,4) = mg_data_test(t);
%     test_output(j) = mg_data_test(t+1); 
%     
%     j = j +1;
%     
% end

%Dados de entrada (t-18)(t-12)(t-6)(t) Saída (t+6)
%Gerando dados

j=1;
for t=19:1044
    
    input(j,1) = mg_data_training(t-18);
    input(j,2) = mg_data_training(t-12);
    input(j,3) = mg_data_training(t-6);
    input(j,4) = mg_data_training(t);
    output(j) = mg_data_training(t+6); 
    
    j = j +1;
end
% % Teste

j = 1;
for t = 19:444
    test_input(j,1) = mg_data_test(t-18);
    test_input(j,2) = mg_data_test(t-12);
    test_input(j,3) = mg_data_test(t-6);
    test_input(j,4) = mg_data_test(t);
    test_output(j) = mg_data_test(t+6); 
    
    j = j +1;
    
end


%Treinamento
[epoch, erro] = ffnn(input,output, test_input, test_output, 0.2,0,0.01, 400, 1);
%
%plot(1:epoch, erro, '-')