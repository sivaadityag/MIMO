clc;
clear all;

N = 10;
N_sim = 1;

EGT_gain = 0;
PSK_gain = 0;

for j = 1:N_sim
    h = (1/sqrt(2))*(randn(1,N)+1i*randn(1,N));
    [U,S,V] = svd(h);
    w_opt = V(:,1);

    EGT_gain = EGT_gain + 10*log10( (norm(h, 1)^2)/N ); 

    M = 8;

    psk_vec = zeros(N,1);
    % 
    % for i=1:N
    %     theta_rad = angle(w_opt(i));
    % 
    %     if theta_rad>=0
    %         psk_theta = floor(M*(theta_rad/2*pi) + 1/2);
    %     else
    %         psk_theta = floor((M+1)/2) + floor( (M*(pi+theta_rad)/(2*pi)) + 1/2);
    %     end
    % 
    %     psk_vec(i) = (cos(psk_theta* (2*pi/M)) + 1i* sin(psk_theta*(2*pi/M)))/sqrt(N);
    % 
    % end
    % PSK_gain = PSK_gain + 10 * log10( (norm(h*psk_vec, 2)^2) );

    psk_vec_blk = decode(w_opt, M);

    for i=1:N
        psk_vec(i) = exp(2*pi*1i*psk_vec_blk(i)/M)/sqrt(N);
    end
    PSK_gain = PSK_gain + 10 * log10( (norm(h*psk_vec, 2)^2) );
 end

% w_opt = [1+0.2i, 1-0.2i, -1+0.2i];
% M = 2;
% psk_vec_blk = decode(w_opt, M);

function g = decode(x, M)
    eta = exp(2*pi*1i/M)
    arg = angle(x)*M/(2*pi)
    g = round(arg)
    [~, u] = sort(g-arg)
    
    p = conj(x).*eta.^g
    sum(p)
    p(u)
    v = [sum(p); p(u)*(eta-1)]
    abs(cumsum(v))
    [~, best] = max(abs(cumsum(v)))
    (1:best - 1)
    u((1:best - 1))
    g
    g(u(1:best - 1)) = g(u(1:best - 1)) + 1
    g = mod(g - g(1), M)
end


% % EGT_gain/N_sim
% % PSK_gain/N_sim