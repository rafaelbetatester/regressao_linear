thetao=zeros(5, 1);
custoTotal = 0;
erroRelativo = 0;
M = 100
for ii = 1:M

	%% Limpa e fecha figuras
	%clear all; close all; clc

	%% Carrega os dados
	data = load('dados.txt');
	X = data(:, 1:4);
	y = data(:, 5);
	m = length(y);

	%Randomiza as linhas da matriz
	idx = randperm(m,m);
	X = X(idx,:);
	y = y(idx,:);

	% Normaliza a matriz X;
	[X mu sigma] = featureNormalize(X);

	% Adiciona a coluna de 1's
	X = [ones(m, 1) X];

	%Constantes do Gradiente Descendente e de validação
	alpha = 0.3;
	num_iters = 100;
	k = 5;
	%%%%%%%%%%%Treinamento
    for jj=1:k
        aux3=[zeros(1,(jj-1)*floor(m/k)),ones(1,floor(m/k)),zeros(1,m-jj*floor(m/k))];
		X2=X(find(aux3),:);
		y2=y(find(aux3),:);
		m2=length(y2);
		%X2=[ones(m2,1) X2];

        % Theta inicial e Gradiente Descendente
        theta = zeros(5, 1);
        [theta, J_history] = gradientDescentMulti(X2, y2, theta, alpha, num_iters);
        thetao = thetao + theta/k;
        %%%%%%%%%%%Validação
        clear X2;
        clear y2;
		clear m2;
		X2=X(find(not(aux3)),:);
		y2=y(find(not(aux3)),:);
		m2=length(y2);
		%X2=[ones(m2,1) X2];
		CUSTO = computeCostMulti(X2,y2,theta);
		CUSTO = sqrt(CUSTO);
		custoTotal = CUSTO/k + custoTotal;
		CUSTO;
		theta;
		diferenca = X2*theta - y2;
		t=length(y2);
		soma=0;
        for i = 1:t
            a=diferenca(i)/y(i);
            if (a<0)
                a=-a;
            endif
            soma = soma + a;
        endfor
        soma=100*soma/t;
        erroRelativo = erroRelativo + soma/k;
		clear X2;
		clear y2;
		clear m2;
    endfor
endfor
thetao = thetao/M
erroRelativo = erroRelativo/M
custoTotal = custoTotal/M
