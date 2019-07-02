%uninstall
disp('Uninstalling...')
current = pwd;
rmpath(fullfile(current,'utils'));
rmpath(fullfile(current,'RGBuvHistBlock'));
savepath
disp('Done!');
