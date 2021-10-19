% Digital modulations under SNR equal to 
% [-10; 0; 10; 25] used for the research assignment.
% Generated mods: 8PAM, 16APSK, QPSK, 16PSK, 16QAM, 64QAM
% Antonios Antoniou - aantonii@ece.auth.gr
% Anestis Kaimakamidis - anestisk@ece.auth.gr
% 2021 Aristotle University Thessaloniki 

pkg load communications

% Number of symbols M and varying SNR in an AWGN channel.
M = [2 4 8 16 64];
SNR = [-10 0 10 25];

% Produce 100 random samples depending on M.
for i = 1:5
  data(i, :) = randi([0 M(i)-1], 100, 1)'; 
endfor

figure = figure()

% 8PAM modulation.
for i = 1:4
  pam_data = pammod(data(3, :), M(3));
  pam_noise = awgn(pam_data, SNR(i), 'measured');
  subplot(2, 2, i);
  scatter(real(pam_noise), imag(pam_noise), 9, "r", "filled");
  title(sprintf ("SNR: %d", SNR(i)));
endfor
print(figure, "latex/8pam.png", "-S872,654")
clf
  
  
% 16APSK modulation with an external 8PSK constellation,
% a 4PSK constellation with no initial phase in the middle,
% and a 4PSK one with pi/4 initial phase internally.
for i = 1:4
  apsk_data1 = 5 * pskmod(data(3, :), M(3));
  apsk_data2 = 3 * pskmod(data(2, :), M(2));
  apsk_data3 = pskmod(data(2, :), M(2), pi/4);
  apsk_data = [apsk_data1 apsk_data2 apsk_data3];
  apsk_noise = awgn(apsk_data, SNR(i), 'measured');
  subplot(2, 2, i);
  scatter(real(apsk_noise), imag(apsk_noise), 9, "r", "filled");
  title(sprintf ("SNR: %d", SNR(i)));
endfor
print(figure, "latex/16apsk.png", "-S872,654")
clf


% QPSK modulation with no initial phase.
for i = 1:4
  qpsk_data = pskmod(data(2, :), M(2));
  qpsk_noise = awgn(qpsk_data, SNR(i), 'measured');
  subplot(2, 2, i);
  scatter(real(qpsk_noise), imag(qpsk_noise), 9, "r", "filled");
  title(sprintf ("SNR: %d", SNR(i)));
endfor
print(figure, "latex/qpsk.png", "-S872,654")
clf

  
% 16PSK modulation with initial phase theta = pi/4.
for i = 1:4
  psk_data = pskmod(data(4, :), M(4), pi/4);
  psk_noise = awgn(psk_data, SNR(i), 'measured');
  subplot(2, 2, i);
  scatter(real(psk_noise), imag(psk_noise), 9, "r", "filled");
  title(sprintf ("SNR: %d", SNR(i)));
endfor
print(figure, "latex/16psk.png", "-S872,654")
clf

 
% 16QAM modulation
for i = 1:4
  qam_data = qammod(data(4, :), M(4));
  qam_noise = awgn(qam_data, SNR(i), 'measured');
  subplot(2, 2, i);
  scatter(real(qam_noise), imag(qam_noise), 9, "r", "filled");
  title(sprintf ("SNR: %d", SNR(i)));
endfor
print(figure, "latex/16qam.png", "-S872,654")
clf

% 64QAM modulation
for i = 1:4
  qam_data = qammod(data(5, :), M(5));
  qam_noise = awgn(qam_data, SNR(i), 'measured');
  subplot(2, 2, i);
  scatter(real(qam_noise), imag(qam_noise), 9, "r", "filled");
  title(sprintf ("SNR: %d", SNR(i)));
endfor
print(figure, "latex/64qam.png", "-S872,654")