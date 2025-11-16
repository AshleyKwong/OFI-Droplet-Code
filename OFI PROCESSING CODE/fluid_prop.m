function [rho_air, visc_air,visc_oil, rho_w, g, P_atm] = fluid_prop(T_atm, P_atm);

% This program computes fluid properties of air and water from the inputed
% atmospheric conditions.
% Temperature given in Celsius
% Pressure given in mmHg
% Philippe Lavoie 28-07-2003
% Revised: Ron Hanson, convert pressure
% Revised: T. Medjnoun 04 08, 2017, estimation of the silicone 50 cst oil viscosity

P_atm = P_atm*0.75006375541921;     % Convert to millimeter of mercury

rho_w4  = 1000;         % density of water at 4oC in kg/m^3
R_air   = 287.058;      % j/kg K
phi     = 50.9097;      % polar latitude at Southampton, UK
H       = 60;           % average altitude around Bolderwood Campus
g       = 9.780327*(1 + 0.0053024*sin(2*phi) - 0.0000058*sin(2*phi)^2 - 0.000003088*H); % UK's National Physical Laboratory formulat
C_air = 1.458e-6;       % Curve fit constant from thermodynamics
S_air = 110.4;          % Another curve fit constant!
a1 = 583.63;            % This is the coefficient for a quick curve fit made based on the thermodynamic data foun in 'Fundamental of Heat and Mass Transfer', 4th Ed. by Incropera and DeWitt
a2 = 3.0514;            % same as a1
a3 = -0.0056;           % same as a1
SG_Hg = 13.6 - 0.0024 * T_atm;  %Specific gravity of Hg

% Coefficients from the second order polynomial curve fit for the viscosity 
% correction, these coefficient have been obtained from a calibration of
% the silicone oil viscosity against a wide range of temperature [10 deg to
% 40 deg] at Southampton University, using DXR Rheometer - 

p1 = 1.738e-05;         % Coefficients from 2016
p2 = -0.00174;
p3 = 0.08225;

p1 = 1.334e-05;         % Coefficients from 2021
p2 = -0.001579;
p3 = 0.07617;

% visc_oil = mu0*exp(-b*T_atm); % exponential model for temperature correction of the oil viscosity

visc_oil = p1*T_atm^2+p2*T_atm+p3; % output is dynamic viscosity in Pa*s 

P_atm= rho_w4 * SG_Hg * P_atm * g;  % Give pressure in Pascals
T_atm = T_atm + 273.15;      % Changes temperature to Kelvin

rho_air = P_atm / (R_air * T_atm);
visc_air = C_air * T_atm^(3/2) / (S_air + T_atm);
rho_w = a1 + a2 * T_atm + a3 * T_atm^2;


