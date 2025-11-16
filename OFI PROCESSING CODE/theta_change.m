function theta_new = theta_change(theta_i, z_height, searchBoxXcoords, imageXrange, mmperpixfactor)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% make sure everything is in mm.
searchBoxcenter = ((searchBoxXcoords(1) + searchBoxXcoords(3))/2)*mmperpixfactor;
imageCenter= ((imageXrange)/2)*mmperpixfactor; % gives the relative 

delta_x = (searchBoxcenter - imageCenter); % this is the offset from the center of the image where theta from davis is defined.
term2 = delta_x + z_height/(tand(theta_i));
theta_new = atand(z_height / term2);
end