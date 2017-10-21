%Rede Neural FeedForward para predição temporal
% 1 Camada Oculta com n neurônios e 1 camada de saída com neurônio unico


function [epoch, erro] =  ffnn(input,output,test_input,test_output, P, Q, R, epoch, learningRate)


%Heurísticas para tanh
a = 1.72;
b = 1;
%adicionando o bias

input(:,end +1) = 1.0;
nSample = length(input(1,:));
nHiddenLayer = 10;
nOutputLayer = 1;





w_hidden = rand(nHiddenLayer, nSample);
w_out = rand(1, nHiddenLayer + 1);
nW = numel(w_hidden) + numel(w_out);

matrixP = P*eye(nW);
matrizQ = Q*eye(nW);


for n = 1:epoch
%propagando a entrada na rede (feedforward)
    uO = [];
    for i= 1:length(input(:,1))
        uH = [];
        
        for j = 1:nHiddenLayer
            soma = dot(w_hidden(j,:),input(i,:));
            uH(j) = a*tanh(b*soma);
        end

        uH(end+1) = 1;
        soma = dot(w_out,uH);
        uO(i) = a*tanh(b*soma);



        %Filtro de Kalman

        %Calculando o jacobiano

        for x = 1:length(uH)-1
            dfUh(x) = dF(uH(x));

        end
        
        dfUo = dF(uO(i));
        
        
     
        

        D = w_out(:,end -1).*dfUh;

      
        outer=[];
        for x=1:length(D)
            for y=1:length(input(i,:))-1
                outer(x,y) = D(x)*input(i,y);
            end
        end

        outer(:,end+1) = D;
        
        outer = reshape(outer,[1,numel(w_hidden)]);
        
        H = [outer uH];
        
        

        S = H*matrixP*H' +R;


        K  = matrixP*H'*inv(S);

        dW = K*(output(i) - uO(i));
        dW = dW';
        
        dWHidden = reshape(dW(1:numel(w_hidden)),size(w_hidden));
        dWOut = reshape(dW(numel(w_hidden)+1:end),size(w_out)); 
        

        %Update dos pesos

        
        w_hidden = w_hidden + learningRate*dWHidden;
        w_out = w_out + learningRate*dWOut;
        
        matrixP = matrixP - K*H*matrixP;
        
        if Q ~= 0
            matrixP = matrixP + matrizQ;
            
        end
        
        


    end    
    
    erro(n) = eqmn(output, uO);
    disp(erro(n))
    

end

%Previsão
uO = [];
    for i= 1:length(test_input(:,1))
        uH = [];
        
        for j = 1:nHiddenLayer
            soma = dot(w_hidden(j,:),input(i,:));
            uH(j) = a*tanh(b*soma);
        end

        uH(end+1) = 1;
        soma = dot(w_out,uH);
        uO(i) = a*tanh(b*soma);
        
    end
    
    erro = test_output - uO;
    plot(1:length(test_output), uO, '-',1:length(test_output), test_output, '--')
    %plot(1:length(test_output), erro, '-')


end



function dF = dF(x)
a = 1.72;
b = 1;
f = tanh(b*x);
dF = a*b*(1 - f*f);
end

function erro  = eqmn(yDesejado, y)

N = length(yDesejado);
sumX = sum(y)/N;
sumY = sum(yDesejado)/N;
eqm = sum((yDesejado- y).^2)/N;

erro = eqm;  %/(sumX*sumY);


end



