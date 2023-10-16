% particle_filter.m

% 2023 OCT 09
% Tershire

% based on Dr. Briere, Yves @ISAE-SUPAERO 

close all
clear

%% setting ////////////////////////////////////////////////////////////////
K_max = 100; % max number of time steps
do_pause = false;

% noise ===================================================================
Q = [0.001 0
     0 0.001]; % state
R = [0.04 0
     0 0.04]; % measurement

% particle ================================================================
N = 10; % number of particles
 
% space ===================================================================
x_min = -2;
x_max = 2;
y_min = -2;
y_max = 2;

% initial =================================================================
x_true = [1, 0]';

% distribution
x_bar = x_true + [-0.0, 0.0]'; % initial guess
% x_hat = x_true;

P = [0.2 0
     0 0.2]; % variance

% particle
omega = 1/N*ones(1, N); % weight
x = x_bar + chol(P)*randn(2, N); % particles
 
% draw --------------------------------------------------------------------
figure(1); % particle, real positions, measurement, ellipsoid of confidence
axis equal;
axis([x_min, x_max, y_min, y_max]);
hx = line(x_true(1), x_true(2), 1, 'Marker', 'o', 'color', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
hy = line(x_true(1), x_true(2), 1, 'Marker', 'x', 'color', 'r', 'MarkerSize', 12, 'LineWidth', 2);
hold on;
% hz = scatter(x(1, :), x(2, :), 40, [0.1, 0, 0.9], 'filled');
hz = scatter(x(1, :), x(2, :), 40, [0.1, 0, 0.9]);
hz.MarkerFaceColor = [0.1, 0, 0.9];
hz.MarkerFaceAlpha = 'flat';
hz.AlphaDataMapping = 'none';
hz.AlphaData = omega/max(omega);

% performance xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ses = [];
variances = [];
ks_resample = [];

% centroid = compute_centroid(x, N)
% hc = line(centroid(1), centroid(2), 1, 'Marker', '*', 'MarkerFaceColor', 'g');

com = compute_CoM(x, omega);
hm = line(com(1), com(2), 1, 'Marker', '^', 'color', 'g', 'MarkerFaceColor', 'g');
% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% create and plot ellipsoid
X0 = sum(x.*[omega; omega], 2);
M = (x.*omega - X0*ones(1,N).*omega)*(x.*omega - X0*ones(1,N).*omega)'*N;
[U, S, V] = svd(M);
% a = 0:0.1:2*pi;
a = linspace(0, 2*pi, 20);
sigma = [sqrt(S(1, 1))*cos(a); sqrt(S(2, 2))*sin(a)];
variance = det(sigma*sigma');
ell0 = sigma*3; % 3-sigma interval,
% corresponds to 98.8% of confidence interval (well explained in J.Sola Phd Thesis)
ell = X0*ones(1, length(a)) + V'*ell0;
hxell = line(ell(1, :), ell(2, :), 'LineWidth', 1.5, 'Color', 'm');
figure(2);  %Particles weights (sorted)
hstem = stem(sort(omega));
% -------------------------------------------------------------------------

%% main ///////////////////////////////////////////////////////////////////
for k = 1:K_max
    if do_pause
        pause;
    end

    % state ===============================================================
    % motion model
    u = [-2*pi/K_max*sin(2*pi*k/K_max)
          2*pi/K_max*cos(2*pi*k/K_max)];
    
    w = chol(Q)*randn(2,1);
    x_true = x_true + u + w;

    % measurement =========================================================
    v = chol(R)*randn(2,1);
    z_true = x_true + v;
    
    % draw ----------------------------------------------------------------
    hx.XData = x_true(1);
    hx.YData = x_true(2);
    hy.XData = z_true(1);
    hy.YData = z_true(2);
    
    % performance xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%     hc.XData = centroid(1, :);
%     hc.YData = centroid(2, :);
    
	hm.XData = com(1, :);
    hm.YData = com(2, :);   
    % xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    title(['step ', num2str(k)]);
    if do_pause
        pause;
    end
    title(['step ', num2str(k), ' : Predicting']);
    if do_pause
        pause(0.01);
    end
    % ---------------------------------------------------------------------

    % predict =============================================================
    for i = 1:N
        % x
        w = chol(Q)*randn(2, 1);
        x(:, i) = x(:, i) + u + w;
    end
    
    % resample XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    N_eff = 1/(omega*omega');
    if N_eff < 0.35*N
        [omega, x] = resample(omega, x);
        
%         disp("RESAMPLING")
        ks_resample = [ks_resample, k];
    end
    % XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    % draw ----------------------------------------------------------------
    % update particle positions after prediction
    hz.XData = x(1, :);
    hz.YData = x(2, :);
    
    % performance xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%     centroid = compute_centroid(x, N)
%     hc.XData = centroid(1, :);
%     hc.YData = centroid(2, :);
    
    com = compute_CoM(x, omega);
	hm.XData = com(1, :);
    hm.YData = com(2, :);
    % xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    % update ellipsoid after prediction
    X0 = sum(x.*[omega; omega], 2);
    M = (x.*omega - X0*ones(1, N).*omega)*(x.*omega - X0*ones(1, N).*omega)'*N;
    [U, S, V] = svd(M);
    % a = 0:0.1:2*pi;
    a = linspace(0, 2*pi, 20);
    sigma = [sqrt(S(1, 1))*cos(a); sqrt(S(2, 2))*sin(a)];
    variance = det(sigma*sigma');
    ell0 = sigma*3;
    ell = X0*ones(1, length(a)) + V'*ell0;
    set(hxell, 'Xdata', ell(1, :), 'Ydata', ell(2, :));
    figure(1); title(['step ', num2str(k), ' ; After Prediction']);
    if do_pause
        pause;
    end
    % ---------------------------------------------------------------------

    % update ==============================================================
    for i = 1:N
        % omega
        omega(i) = omega(i)*exp(-1/2*(x(:, i) - z_true)'*R^-1*(x(:, i) - z_true));
    end
    omega = omega/sum(omega); % normalize
    
    % draw ----------------------------------------------------------------
    % update particle weights and ellipsoid of confidence
    hz.XData = x(1, :);
    hz.YData = x(2, :);
    hz.AlphaData = omega/max(omega);
    hz.MarkerFaceAlpha = 'flat';
    hstem.YData = sort(omega);
    
    % performance xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
%     centroid = compute_centroid(x, N)
%     hc.XData = centroid(1, :);
%     hc.YData = centroid(2, :);
    
    com = compute_CoM(x, omega);
	hm.XData = com(1, :);
    hm.YData = com(2, :);
    % xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    % update Ellipsoid
    X0 = sum(x.*[omega; omega], 2);
    M = (x.*omega - X0*ones(1, N).*omega)*(x.*omega - X0*ones(1, N).*omega)'*N;
    [U, S, V] = svd(M);
    % a = 0:0.1:2*pi;
    a = linspace(0, 2*pi, 20);
    sigma = [sqrt(S(1, 1))*cos(a); sqrt(S(2, 2))*sin(a)];
    variance = det(sigma*sigma');
    ell0 = sigma*3;
    ell = X0*ones(1, length(a)) + V'*ell0;
    set(hxell, 'Xdata', ell(1, :), 'Ydata', ell(2, :));
    figure(1); title(['step ', num2str(k), ' ; After Update']);
    figure(2); title(['step ', num2str(k), ' ; After Update ; N_eff = ', num2str(N_eff)]);
    if do_pause
        pause;
    end
    % ---------------------------------------------------------------------
    
    % performance evaluation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    se = compute_se(x_true, com);
    ses = [ses, se];
    variances = [variances, variance];
    % xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
end

%% resampling algorithm
function [W, X] = resample(W, X)
% Resampling algorithm
% Multinomial algorithm
% W : weights of the particles
% indx : index of the particles to be resampled
    W = W/sum(W); % just in case the weights are not normalized
    N = length(W);
    Q = cumsum(W);
    i = 1;
    indx = 1:N; % Preallocate indx (save time)
    while (i <= N)
        sampl = rand(1, 1);
        j = 1;
        while (Q(j) < sampl)
            j = j + 1;
        end
        indx(i) = j;
        i = i + 1;
    end
    X = X(:, indx);
    W = ones(1, N)/N;
end

%% performance metrics
function centroid = compute_centroid(x, N)
    centroid = (1/N)*sum(x, 2);
end

function com = compute_CoM(x, omega)
    mass = sum(omega);
    com = sum(x.*omega, 2)/mass;
end

function se = compute_se(x_true, com)
    error = x_true - com;
    se = error'*error;
end
