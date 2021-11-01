% Generates samples of: 8PAM, BPSK, QPSK, 16PSK, 16QAM, 64QAM.
% SNR values are set from -5 to 10 with a step of 1.
% These samples are used as the training set for a Deep Neural Network.
% Antonios Antoniou - aantonii@ece.auth.gr
% Anestis Kaimakamidis - anestisk@ece.auth.gr
% 2021 Aristotle University Thessaloniki 

pkg load communications

M = [2 4 8 16 64];
SNR = -5:1:10;

% Set to 3500 for the training set. Produces 105000 samples.
iterations = 100; 
ratios = length(SNR)
% Number of samples per modulation.
% We use this to skip ahead in the 'samples' array. 
interval = iterations * ratios;

% A table of the modulations used for reference
mods = [ "BPSK"; "QPSK"; "8PSK"; "16QAM"; "64QAM"];

% The index used to signify what modulation we are
% currently generating, based on the 'mods' array.
% We add this index before every sample as a label
% that the network can use while training. 
i = 0; 

% BPSK samples.
for j = 1:iterations
  for k = 1:ratios
    data = randi([0 M(1)-1], 1000, 1);
    modulated = pskmod(data, M(1));
    noise = awgn(modulated, SNR(k));
    samples(interval*i + ratios*(j-1) + k, :) = [i+1 SNR(k) noise];
  endfor
endfor
"End of BPSK."


% QPSK samples.
i = i + 1;
for j = 1:iterations
  for k = 1:ratios
    data = randi([0 M(2)-1], 1000, 1);
    modulated = pskmod(data, M(2));
    noise = awgn(modulated, SNR(k));
    samples(interval*i + ratios*(j-1) + k, :) = [i+1 SNR(k) noise];
  endfor
endfor
"End of QPSK."


% 8PSK samples.
i = i + 1;
for j = 1:iterations
  for k = 1:ratios
      data = randi([0 M(3)-1], 1000, 1);
    modulated = pskmod(data, M(3));
    noise = awgn(modulated, SNR(k));
    samples(interval*i + ratios*(j-1) + k, :) = [i+1 SNR(k) noise];
  endfor
endfor
"End of 8PSK."


% 16QAM samples.
i = i + 1;
for j = 1:iterations
  for k = 1:ratios
    data = randi([0 M(4)-1], 1000, 1);
    modulated = qammod(data, M(4));
    noise = awgn(modulated, SNR(k))';
    samples(interval*i + ratios*(j-1) + k, :) = [i+1 SNR(k) noise];
  endfor
endfor
"End of 16QAM."


% 64QAM samples.
i = i + 1;
for j = 1:iterations
  for k = 1:ratios
    data = randi([0 M(5)-1], 1000, 1);
    modulated = qammod(data, M(5));
    noise = awgn(modulated, SNR(k))';
    samples(interval*i + ratios*(j-1) + k, :) = [i+1 SNR(k) noise];
  endfor
endfor
"End of 64QAM."


csvwrite("training_set.csv", samples);
"CSV file generated and saved."
