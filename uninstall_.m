%uninstall
disp('Uninstalling...')
current = pwd;
rmpath(fullfile(current,'utils'));
savepath
disp('Done!');
