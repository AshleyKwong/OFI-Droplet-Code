% Ashley Kwong 
% July 2024
% OFI processing code
%% Preamble for plotting
clear; clc; close all;
% Set default font and font size for axes and text
set(groot, 'defaultAxesFontName', 'Cambria Math');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontName', 'Cambira Math');
set(groot, 'defaultTextFontSize', 12);
% Set default color order for axes (example: MATLAB's default, or customize)
set(groot, 'defaultAxesColorOrder', [0 0 0; 1 0 0; 0 0 1]);

%% Adding necessary paths...
%addpath of the readimx function
addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB\Experimental Campaign 1\OFI\readimx-v2.1.9-win64'); 
addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB\ErrorBarFormat'); 
addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB'); 
addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB\Experimental Campaign 1\OFI'); 
addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB') % for the parse error
addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB\Experimental Campaign 1\OFI\OFI Codes Tak'); 


%% initial pre check 
close all; clc; 
% imageDir ='D:\TIF_y235_aoan08_aoafn12_secondplate_U20_01\B0200.tif'; % the specific case to analyze
% imageDir ='D:\y235_aoan08_aoafn12_U20_downstream_02\B0200.tif'; % the specific case to analyze
% imageDir ='D:\TIF y235_aoan11_aoafn11_U20_downstream_02\B0200.tif'; % the specific case to analyze
imageDir = 'D:\TIF y235_aoan11_aoafn11_U20_secondplate_01\B0200.tif'; 

% imageDir = 'D:\TIF EmptyTunnel_downstream_U20_2\B0200.tif'; % the specific case to analyze
% imageDir = 'D:\TIF EmptyTunnel_secondplate_U20\B0200.tif'; 
% imageDir = 'D:\TIF Case 2 Downstream\B0200.tif';
% imageDir = 'D:\TIF Case 2 Second Plate\B0200.tif'; 


% tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06062025\y235_aoan08_aoafn12\y235_aoan08_aoafn12_U20_downstream_02.mat'; 
% tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06052025\y235_aoan11_aoafn11_repeat\y235_aoan11_aoafn11_U20_downstream_02.mat'; 
% tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06062025\y250_aoan04_aoafn06\y250_aoan04_aoafn06_U20_downstream_02.mat'; 
% tunnelConditions =  'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06042025\Empty Tunnel\Emptytunnelofi_U20_downstream_02.mat'; 

% tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06072025\y235_aoan08_aoafn12\y235_aoan08_aoafn12_U20_secondplate_01.mat'; % tunnel conditions at time of measurement.
% tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06082025\emptytunnel_secondplate\emptytunnel_secondplate_U20_secondplate_01.mat'; 
% tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06072025\y250_aoan04_aoafn06\y250_aoan04_aoafn06_U20_secondplate_02.mat';
tunnelConditions = 'F:\LAB7 COMPUTER\OFI NIDAQ PRESSURES\AK_OFI\06072025\y235_aoan11_aoafn11\y235_aoan11_aoafn11_U20_secondplate_01.mat'; 

% calibrationFile = 'D:\ALK_OFI_250530_105600\Properties\Calibration\Calibration.xml'; 
calibrationFile = 'D:\ALK_OFI_06072025\Properties\Calibration\Calibration.xml'; 
folder = 'D:\TIF y235_aoan11_aoafn11_U20_secondplate_01\'; 
orgimg = imread(imageDir);
%------------ USER INPUT -------------------------------------------
numOfCameras = 4;
pixelpermm =  14.95;  % [15.95, 14.95, 14.95, 14.95, 14.95]; 
cameraNo = 2; 
calibration_cameraNo= 3; 
cam_z = 476.349; % vertical distance between the camera and calibration plate. 
camD       =  43.85;  %43.85; %45.49;   % Camera incident angle with respect the normal of measurement plane (deg)
% first plate calibration
% cam 1 = 48.10
% cam 2 = 44.48
% cam 3 = 46.63
% cam 4 = 45.49
% cam 5 = 48.27

% second plate calibration
% cam 1 = 48.10 (this camera was not used)
% cam 2 =  35.36 
% cam 3 = 43.85
% cam 4 = 44.97 
% cam 5 = 49.69

%---------------------------------------------------------
cameraImg = struct(); 
figure(); imagesc(orgimg); 

for i = 1:numOfCameras
    % need to define each camera
    hold on; 
    roi = drawrectangle;
    pos = roi.Position;
    cameraImg(i).orgImageCoord = pos; % this is the selected coordinates in the global image.
    cameraImg(i).userDefinedImage = orgimg(pos(2):pos(2)+pos(4), pos(1): pos(1)+pos(3));
    hold off;
end 

cameraImg = getCameras(cameraImg); % redefining the struct as one w the global coordinate system etc. this is where the globalCoords get defined!

calibrationXML = parseCameraXML(calibrationFile); % grab the calibration !  

%-----------------------------Physical Coordinate setting
% MAKE SURE THE CAMERAS AND THE CAL CAMERA IDX LINES UP! 
x0 = (cameraImg(1).GlobalCoordinatesFilteredImg(4) - calibrationXML(2).OriginPixelPosition(1))*calibrationXML(2).Scales.X.FactorMmPerPixel; % from the rhs --> until 0 point from first cam in the struct
xf = (cameraImg(end).GlobalCoordinatesFilteredImg(3) - calibrationXML(5).OriginPixelPosition(1))*calibrationXML(5).Scales.X.FactorMmPerPixel; % from the rhs --> until 0 point. 
totalMeasurementWindow = x0 - xf;     

% physical offset downstream of where the 0 point is from tunnel start. 
% 8400 is based off the start of the second section from which everything
% was measured.
x_globalStart = -(8400 + x0)%-7311; %0; %(x0); % -( x0); % the location of the calibration origin. evt is measured relative to this. negative to go w direction of streamwise. 

%% NEED TO ADD A LOOP FOR ALL THE CAMERAS! 

d = dir(folder);
dirnames = {d(~[d.isdir]).name}';

% this gives the frame of the specific camera with some bound protection so
% we dont get the edge of the image. 
streamwise_padding = 0;  
spanwise_padding = 0; 
fullrow_start = cameraImg(cameraNo).GlobalCoordinatesFilteredImg(1) + streamwise_padding + 200; 
fullrow_end = cameraImg(cameraNo).GlobalCoordinatesFilteredImg(2) - streamwise_padding   ; 
fullcol_start =  cameraImg(cameraNo).GlobalCoordinatesFilteredImg(3) + spanwise_padding; 
fullcol_end = cameraImg(cameraNo).GlobalCoordinatesFilteredImg(4) - spanwise_padding; 


img = imread(imageDir, 'PixelRegion', {[fullrow_start ...
        , fullrow_end], ...
        [fullcol_start...
        , fullcol_end]}); % the x and y are shifted

% img = imread(imageDir, 'PixelRegion',{[600, 3100], [1.7e+04,21079]}); 
grayimg = mat2gray(img);
figure(); imagesc(grayimg); 

% mean_I = nan(length(dirnames), 1); 
% for i = 1:100
%     tif_file = fullfile(folder, dirnames{i}); 
    % img = imread(tif_file, 'PixelRegion',{[cameraImg(cameraNo).GlobalCoordinatesFilteredImg(1) ...
    %     , cameraImg(cameraNo).GlobalCoordinatesFilteredImg(2)], ...
    %     [cameraImg(cameraNo).GlobalCoordinatesFilteredImg(3) ...
    %     , cameraImg(cameraNo).GlobalCoordinatesFilteredImg(4)]});  % gets each CAMERA frame. 

%     grayimg_current= mat2gray(img);
%     mean_I(i) = mean(grayimg_current(:)); 
% end
% tavg_meanI = mean(mean_I); 
% grayimg_meansub = grayimg - tavg_meanI; 
grayimg_meansub = grayimg; 


%%
threshold = 0.3; 
bw = edge(grayimg_meansub, 'canny', threshold); %By using two thresholds, the Canny method is less likely than the other methods to be fooled by noise, and more likely to detect true weak edges.
figure();

imshow(bw); 
% Find the row and column indices of edge pixels
[edge_rows, edge_cols] = find(bw);
X = [edge_rows, edge_cols]; 
gapWidth = 10;  
vertGap = 20; 

% Create a combined structuring element that bridges both horizontally and vertically
seH = strel('line', gapWidth+1, 0);
seV = strel('line', vertGap+1, 90);
combinedNeighborhood = seH.Neighborhood | seV.Neighborhood; % logical OR
seCombined = strel('arbitrary', combinedNeighborhood);

BW_closed = imdilate(bw, seCombined);
figure(); 
imshow(BW_closed); 

%%
CC = bwconncomp(BW_closed); % or use label matrix L
props = regionprops(CC, 'Area', 'Centroid', 'BoundingBox', 'Orientation');

for k = 1:length(props)
    fprintf('Region %d: Area=%.1f, Centroid=(%.1f, %.1f)\n', k, props(k).Area, props(k).Centroid(1), props(k).Centroid(2));
end
labeledImage = labelmatrix(CC);
RGB_label = label2rgb(labeledImage, 'jet', 'k', 'shuffle'); % colorful display

figure; 
imshow(RGB_label);
title(sprintf('Connected Components Colored (Total: %d)', CC.NumObjects));

%%
minArea = 500; % pixels
largeClusters = props([props.Area] >= minArea);

widths = arrayfun(@(s) s.BoundingBox(3), largeClusters);
sortWidth = find(widths > 500);
largeClusters = largeClusters(sortWidth);
heights = arrayfun(@(s) s.BoundingBox(4), largeClusters);
sortHeight = find(heights > 50); 
largeClusters = largeClusters(sortHeight);

figure()
imshow(grayimg); hold on;
hRects = gobjects(length(largeClusters), 1); % preallocate handles

for k = 1:length(largeClusters)
    % rectangle('Position', largeClusters(k).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 1);
    plot(largeClusters(k).Centroid(1), largeClusters(k).Centroid(2), 'r*');
    x2 = largeClusters(k).BoundingBox(1) + largeClusters(k).BoundingBox(3) - 10; % keep it 5 pixels away from the edge
    x1_new= largeClusters(k).BoundingBox(1) + (largeClusters(k).BoundingBox(3))/1.5 ; % a little bit less than half for the bounding box based on the oil drop spread
    width_new = x2 - x1_new;
    y1_new = largeClusters(k).Centroid(2) - 10;
    search_box = [x1_new, y1_new,  width_new, 20];
    largeClusters(k).searchBox = search_box; % the search box is in the units of the filtered image. 

    rectangle('Position', search_box, 'EdgeColor', 'g', 'LineWidth', 1);

    % Create interactive rectangle if we want to adjust the bounding box. 
    % Prompt the user: adjust or skip?
    userChoice = questdlg(...
        sprintf('Do you want to adjust Box %d?', k), ...
        'Adjust Rectangle', ...
        'Adjust', 'Skip', 'Skip');

    if strcmp(userChoice, 'Adjust')
        hRects(k) = drawrectangle('Position', search_box, ...
            'Color', 'g', ...
            'LineWidth', 1);
        % Wait for user adjustment, then save the new position
        uiwait(msgbox('Adjust box, then click OK to continue.'));
        largeClusters(k).searchBox = hRects(k).Position;
        largeClusters(k).Centroid(1) = (largeClusters(k).searchBox(1) + (largeClusters(k).searchBox(1) + largeClusters(k).searchBox(3))) /2; % should be the row wise
        largeClusters(k).Centroid(2) = (largeClusters(k).searchBox(2) + (largeClusters(k).searchBox(2) + largeClusters(k).searchBox(4))) /2; 
    else
        % Just display static rectangle
        rectangle('Position', search_box, 'EdgeColor', 'g', 'LineWidth', 1);
        % hRects(k) = []; % No interactive handle needed
    end

end
hold off;
axis on; 
figure();
imshow(grayimg); hold on;
% ---- After user interacts, save new rectangle positions ----
% User must manually adjust rectangles, then run this code to update
for k = 1:length(largeClusters)

    rectangle('Position', largeClusters(k).searchBox, 'EdgeColor', 'g', 'LineWidth', 1);
end

% largeClusters now contains the updated rectangles
hold off; 
% this sets the location of the search boxes based off of whatever image
% frame you want (last or first) -->the bounding box stays constant for the
% whole frame loop
% ---- Optionally allow user to add rectangles ----
addMore = questdlg('Would you like to define additional rectangles?', ...
    'Add Rectangles', 'Yes', 'No', 'No');
if strcmp(addMore, 'Yes')
    moreRects = true;
    while moreRects
        hNewRect = drawrectangle('Color','b','LineWidth',1);
        uiwait(msgbox('Draw a rectangle, adjust it, then click OK.'));
        pos = hNewRect.Position;
        N = length(largeClusters) + 1; % this way things get sorted properly. 
        largeClusters(N).searchBox = pos;
        largeClusters(N).BoundingBox = pos;
        largeClusters(N).Area = pos(3)*pos(4);
        largeClusters(N).Centroid = [pos(1)+pos(3)/2, pos(2)+pos(4)/2];
        
        keepAdding = questdlg('Add another rectangle?', 'Add?', 'Yes','No','No');
        moreRects = strcmp(keepAdding, 'Yes');
    end
end
clf; 
figure();
imshow(grayimg); hold on;
% ---- After user interacts, save new rectangle positions ----
% User must manually adjust rectangles, then run this code to update
for k = 1:length(largeClusters)

    rectangle('Position', largeClusters(k).searchBox, 'EdgeColor', 'g', 'LineWidth', 1);
end

% largeClusters now contains the updated rectangles
hold off; 
%%
% Now perform the fft for the oil drops
% FFT parameters
Fs = 1; % Sampling frequency in Hz
close all; 
oilDropinfo = struct(); 
oilDropinfo.frequency_interp = zeros(0,0); 
oilDropinfo.frequency_uncert = zeros(0,0); 

start_idx = 80  ; 
tic; 
for i = start_idx:length(dirnames)
    tif_file = fullfile(folder, dirnames{i});
    fprintf('Processing Drop: %s\n', tif_file);
    for k = 1:length(largeClusters)
        dropname = sprintf('Drop %d', k);
        oilDropinfo(k).Name = dropname;
         % Get just the image slice of predefined oil drop. box is constant size
         % for now...
         % ------- setting in the context of the global coordinates.
         % formatted to have each vertice. 
         y1 = max(1, round(largeClusters(k).searchBox(2))) + fullrow_start - 1;
         y2 = round(largeClusters(k).searchBox(2) + largeClusters(k).searchBox(4)) + fullrow_start -1 ; 
         x1 = max(1, round(largeClusters(k).searchBox(1))) + fullcol_start -1;
         x2 = round(largeClusters(k).searchBox(1) + largeClusters(k).searchBox(3)) + fullcol_start -1;
        
         % oilDropinfo(k).searchBox = [y1, round(largeClusters(k).searchBox(2) + largeClusters(k).searchBox(4)), x1, round(largeClusters(k).searchBox(1) + largeClusters(k).searchBox(3))]; %org image pixel loc. not global space.
         oilDropinfo(k).searchBox = largeClusters(k).searchBox; % org image pixel loc. 

         oilDropinfo(k).globalSearchBoxRectangularCoord = ([x1, y1  , largeClusters(k).searchBox(3), largeClusters(k).searchBox(4)]); % need to save this in rectangle formatm, x, y , width and height
         oilDropinfo(k).globalSearchBoxCoord = ([x1 , x2, y1, y2]); % this is in absolute --> which is useful later for defining the physical coordinates
         camLoad = imread(tif_file, 'PixelRegion', {[y1, y2], [x1, x2]});
         % Compute average span (along first dimension) and convert to uint16
         % for speeed
         original_N = length(camLoad); % Length of avg_span
         window = hanning(original_N);
         noverlap = floor(length(window))/2;
         N = 2^nextpow2(original_N); % fft more efficient when in power of 2.
         nfft = N; % added zero padding
         % optimize FFT for repeated calls
         fftw('planner','measure');
         freq_slice= nan(size(camLoad,1),1); % clear the freq slice for each large cluster

         for j = 1:size(camLoad, 1)
             avg_detrend = detrend(double(camLoad(j, :))); 
             [pxx, f] = pwelch(avg_detrend, window, noverlap, nfft, Fs); % takes the pwelch
             [~, idx0] = min(abs(f)); % find the mean frequency
             neighborRange = 5;
             idxZero = max(idx0 - neighborRange, 1) : min(idx0 + neighborRange, length(pxx));
             % Zero out those PSD values for the DC signal
             pxx(idxZero) = 0;
             % [~, idx] = max(pxx);
             [~, peakLocforbox, peakLocforboxUncert] = fitGaussian(f, pxx); % single value per box
             freq_slice(j) = peakLocforbox; %(peakLocforbox * Fs / N); 
             % freq_slice(end+1, 2)  = (peakLocforboxUncert* Fs/N); 

         end
         
         oilDropinfo(k).frequency_interp = [oilDropinfo(k).frequency_interp, freq_slice];  % bc matlab is index 1 and bins are index 0 --> need to do -1 - conversion to freq !
         oilDropinfo(k).frequency_uncert= [oilDropinfo(k).frequency_uncert;  std(freq_slice)]; % the stdev as determined by the average of each row. 
         
         oilDropinfo(k).physicalGlobalCoordinates(1:2) = (oilDropinfo(k).globalSearchBoxCoord(1:2) - calibrationXML(calibration_cameraNo).OriginPixelPosition(1)) * calibrationXML(calibration_cameraNo).Scales.X.FactorMmPerPixel; % note the calibration is +1 since we didn't use camera 1. 
         oilDropinfo(k).physicalGlobalCoordinates(3:4) = (oilDropinfo(k).globalSearchBoxCoord(3:4) - calibrationXML(calibration_cameraNo).OriginPixelPosition(2)) * calibrationXML(calibration_cameraNo).Scales.Y.FactorMmPerPixel; 
         % as we go further downstream the coordinates will get more
         % negative.
    end 

end 
endTime = toc; 
fprintf("\n\tTotal time to run : %.2f\n", endTime); 
      %%
t = linspace(start_idx, length(dirnames), (length(dirnames)-start_idx+1)/Fs);

addpath('C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB\Experimental Campaign 1\OFI\OFI Codes Tak');

atm_conditions = load(tunnelConditions);
T_atm = mean(atm_conditions.T);
P_atm = atm_conditions.P0 / 1013; % in atm, the original pressure is reported in what seems like millibar
% P_atm = 0.9928448; % for empty tunnel condition xd=2
P_atm_Pa_uncert = 0.1*1000; % in Pa 
theta      = camD*pi/180;               % Camera angle (rad)
Magnif     = (1/pixelpermm).*10^(-3);    % Camera magnification factor (m/px) check calibration !

% Physical "constants"
sodiumwave = 589.3*10^(-9);             % Low-pressure Sodium wavelength (actually two dominant spectral lines very close together at 589.0 and 589.6 nm)
nair       = 1.0002772;                 % Air refraction index at 25 C (assumed invariant)
noil       = 1.4022;                    % Dow Corning Silicone Oil refraction index at 25 C (assumed invariant)

[rho_air, visc_air,visc_oil, rho_w, g, P_atm_calcd] = fluid_prop(T_atm, P_atm); % reports air in kg/m3
% uncert in T is based off of the std in T --> visc_oil = p1*T_atm^2+p2*T_atm+p3; % output is dynamic viscosity in Pa*s 
T_atm_uncert = std(atm_conditions.T); 
P_dyn_uncert = std(atm_conditions.qTime); 
P_dyn = atm_conditions.qu; 


% ----- Coeffs from fluidprops function, 2021 updated 
p1 = 1.334e-05;         
p2 = -0.001579;
p3 = 0.07617;

R_air   = 287.058;      % j/kg K
%-----------------------------------------------------

x_range = ((cameraImg(cameraNo).GlobalCoordinatesFilteredImg(3:4) - calibrationXML(calibration_cameraNo).OriginPixelPosition(1))*calibrationXML(calibration_cameraNo).Scales.X.FactorMmPerPixel) + x_globalStart; 
y_range = abs ((cameraImg(cameraNo).GlobalCoordinatesFilteredImg(1:2) - calibrationXML(calibration_cameraNo).OriginPixelPosition(2))*calibrationXML(calibration_cameraNo).Scales.Y.FactorMmPerPixel); % note this is rel to the orgin of the cal target, the cal target was centered w tunnel.

RI = imref2d(size(grayimg_meansub), x_range, y_range); 
figure(); imshow(grayimg_meansub); % overplot onto the image
axis on; 
xlabel( 'x [pixel]'); 
ylabel('y [pixel]'); 
% --------------------------------------------------------
hold on; 
for i = 1:length(oilDropinfo)
    oilDropinfo(i).plotXcoordmm = (oilDropinfo(i).physicalGlobalCoordinates(1:2)) + x_globalStart ;
    theta_new = theta_change(camD, cam_z, oilDropinfo(i).plotXcoordmm, x_range);
    theta_rad = theta_new*pi/180;               % Camera angle (rad)
    n0 = sqrt(noil^2-nair^2*(sin(theta_rad))^2);
    mdl_cell = cell(size(oilDropinfo(i).frequency_interp, 1),1);
    tau_specificrow = nan(size(oilDropinfo(i).frequency_interp, 1),1); 
    for k = 1:size(oilDropinfo(i).frequency_interp, 1)
        % where k loops from the pixel row 1 -> k taking the fit through
        % each frame which is the number of columns.
        mdl_pixelrow = fitlm(t, 1./(oilDropinfo(i).frequency_interp(k, :)));
        mdl_cell{k} = mdl_pixelrow;
        tau_specificrow(k)= 2*n0*visc_oil*(mdl_pixelrow.Coefficients.Estimate(2))*Magnif/sodiumwave; % tau uncert is based on visc uncert and the slope uncert, units is Pa. 
    end 
    mdl = fitlm(t, 1./ mean(oilDropinfo(i).frequency_interp)); % gives slope info.

    oilDropinfo(i).thetaNew = theta_new; 
    oilDropinfo(i).modelInfo = mdl_cell;
    oilDropinfo(i).tauPerRow = tau_specificrow; 
    % ----------- SLOPE UNCERTAINITY -------------------------
    % for a certain drop, it gives the max and min possible slope. 
    freq_std =     oilDropinfo(i).frequency_uncert; 
    freq_avg =     1./ mean(oilDropinfo(i).frequency_interp, 'omitnan'); 
    
    [maxstdevval, maxstdevloc] = max(freq_std);
    [minstdevval, minstdevloc] = min(freq_std);
    if maxstdevloc < minstdevloc
        slope1 = ((freq_avg(minstdevloc) - minstdevval) - (freq_avg(maxstdevloc) + maxstdevval)) / (t(minstdevloc) - t(maxstdevloc));
        slope2 = ((freq_avg(minstdevloc) + minstdevval) - (freq_avg(maxstdevloc) - maxstdevval)) / (t(minstdevloc) - t(maxstdevloc));
        % b1 = (freq_avg(maxstdevloc) + maxstdevval) - slope1*t(maxstdevloc);
        % b2 = (freq_avg(maxstdevloc) - maxstdevval) -  slope2*t(maxstdevloc);

    else
        slope1 = ((freq_avg(maxstdevloc) - maxstdevval) - (freq_avg(minstdevloc) + minstdevval)) / (t(maxstdevloc) - t(minstdevloc));
        slope2 = ((freq_avg(maxstdevloc) + maxstdevval) - (freq_avg(minstdevloc) - minstdevval)) / (t(maxstdevloc) - t(minstdevloc));
        % b1 = (freq_avg(minstdevloc) + minstdevval) - slope1*t(minstdevloc);
        % b2 = (freq_avg(minstdevloc) - minstdevval) - slope2*t(minstdevloc);
    end

    
    % --------------------------------------------------------------
    
    [roundedCoeff, roundedError] = significantDigit(mdl.Coefficients.Estimate(2), mdl.Coefficients.SE(2)); 
    slope = roundedCoeff; % pixel/s

    slope_uncert = std([roundedCoeff, slope1, slope2]); % in pixel/s
    oilDropinfo(i).slopeInfo = [slope, slope_uncert]; 

    % ----------Uncertainty Analysis in accordance with Propogation of
    % Uncertainity, see SESA6070 slides
    n0_fixed= sqrt(noil^2-nair^2*(sin(theta))^2); % this is based on the fixed camera angle.
    tau = 2*n0_fixed*visc_oil*slope*Magnif/sodiumwave; % tau uncert is based on visc uncert and the slope uncert, units is Pa. 
    dtau_dTatm = (2*n0*Magnif*slope/sodiumwave)* (2*p1*T_atm + p2);
    dtau_dSlope = 2*n0*visc_oil*Magnif/sodiumwave;
    tau_uncert = sqrt((dtau_dTatm*T_atm_uncert)^2 + (dtau_dSlope*slope_uncert)^2);
    [tau, tau_uncert] = significantDigit(tau, tau_uncert);
    oilDropinfo(i).tau = [tau, tau_uncert]; 

    Cf_uncert = sqrt((dtau_dTatm/P_dyn*T_atm_uncert)^2 + (dtau_dSlope/P_dyn*slope_uncert)^2 + (-2/(P_dyn^2)*tau* P_dyn_uncert)^2);
    Cf = tau / P_dyn;
    [Cf, Cf_uncert] = significantDigit(Cf, Cf_uncert);
    oilDropinfo(i).Cf = [Cf, Cf_uncert]; 

    u_tau = sqrt(tau / (rho_air)); % uncert in this also based off of rho_air --> rho_air = P_atm / (R_air * T_atm);
    dutau_da = 0.5*(u_tau)^(-1.5)*(p1*T_atm^2 + p2*T_atm + p3)*(R_air*(T_atm+273.15))*(2*n0*Magnif)/(sodiumwave*P_atm_calcd);
    dutau_dTatm = 0.5*(u_tau)^(-1.5) * (2*n0*slope*Magnif*R_air/(sodiumwave*P_atm_calcd))*(3*p1*T_atm^2 + 2*p2*T_atm + p3 + 2*p1*T_atm*273.15 + p2*273.15); % need to acct for the fact that rho is found w T in Kelvin
    dutau_dPatm = 0.5*(u_tau)^(-1.5) * (-tau*R_air*(T_atm + 273.15))/((P_atm_calcd)^2);
    u_tau_uncert = sqrt((dutau_dPatm*P_atm_Pa_uncert)^2 + (dutau_dTatm*T_atm_uncert)^2 + (dutau_da*slope_uncert)^2);
    [u_tau, u_tau_uncert, sigDigitTracker] = significantDigit(u_tau, u_tau_uncert);
    oilDropinfo(i).u_tau =[u_tau, u_tau_uncert]; 
    rectangle('Position', largeClusters(i).searchBox, 'EdgeColor', 'g', 'LineWidth', 1);

    formatSpec = sprintf('%%.%df', sigDigitTracker); % e.g. '%.2f'

    % Construct the full LaTeX string with placeholders for tau and tau_uncert
    strFormat = ['$ Drop \\: %i \\: \\tau = %s \\pm %s$ Pa'];

    % Use sprintf with the decimal format specifier on tau and tau_uncert
    str = sprintf(strFormat, i, sprintf(formatSpec, tau), sprintf(formatSpec, tau_uncert));

    % Now create the text with LaTeX interpreter
    text(largeClusters(i).Centroid(1), largeClusters(i).Centroid(2), str, ...
        'Interpreter', 'latex', ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 14, ...
        'FontWeight', 'bold', ...
        'Color', 'k');

end
%% figure for the streamwise dev of Cf
% change desired folder !! 
desiredFolder = 'C:\Users\ak1u24\OneDrive - University of Southampton\MATLAB\Experimental Campaign 1\OFI\Case 6 Streamwise Cf Development\xd=6'; 
cd(desiredFolder); 

figure(); 
hold on; 
xCoord_big = []; 
Cf_big = []; 
Cf_delta = []; 
tau = []; 
u_tau = []; 
 
xd = -8775; %8250; %8775; % mm 

hold on; 
avg_Cf_large = zeros(0, 0);
std_Cf_large = zeros(0,0);
xLoc_large = zeros(0,0);
for i = 1:length(oilDropinfo)

    xCoords = (oilDropinfo(i).physicalGlobalCoordinates(1:2)) + x_globalStart;     
    % xCoords = sort(xCoords + x_globalStart); %  
    xCoord_big(2*i - 1) = xCoords(1);
    xCoord_big(2*i) = xCoords(2);
    oilDropinfo(i).plotXcoordmm = xCoords; % this is the mat file i should save for each run ! 
    Cf_area = oilDropinfo(i).Cf(1);
    if oilDropinfo(i).tau(2) < 0.1 % if the error is unreasonably big
        avg_tau = mean(oilDropinfo(i).tauPerRow);
        std_tau = std(oilDropinfo(i).tauPerRow);
        Cf = [avg_tau, avg_tau]/ P_dyn;
        avg_Cf_large= [avg_Cf_large, Cf];
        Cf_std = [std_tau, std_tau]/P_dyn;
        if size(Cf_std, 2)~= size(Cf_std, 2)
            Cf_std = Cf_std';
        end
        std_Cf_large  = [std_Cf_large,  Cf_std];
        xLoc = oilDropinfo(i).plotXcoordmm; % this is a 2 element array
        xLoc_large = [xLoc_large, xLoc];
        errorbar(xLoc, Cf, Cf_std, "ro", MarkerFaceColor = "r")
    end

end
xlabel("Global x [mm]"); 
ylabel("$C_f$", Interpreter="latex"); 
[~, xLoc_xd]= min(abs(xLoc_large - xd));   
% take a moving mean average of 3 samples
Cf_movingavg = movmean(avg_Cf_large, 3); 
plot(xLoc_large(xLoc_xd),Cf_movingavg(xLoc_xd), "bp", MarkerSize = 10, MarkerFaceColor = "b"); 

fprintf("\tCf at xd = %i : %.6f\n", xd, Cf_movingavg(xLoc_xd)); 
xd6_recheck = struct('Cf', Cf_movingavg(xLoc_xd)); 
fullFilePath2 = fullfile(desiredFolder, 'xd6_recheck');
save(fullFilePath2, 'xd6_recheck'); 
hold off

filename = sprintf('camera%d_recheck_n0adjusted.mat', cameraNo);
fullFilePath = fullfile(desiredFolder, filename);
save(fullFilePath, 'oilDropinfo');
