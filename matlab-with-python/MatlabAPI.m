

pyversion
 

% if count(py.sys.path, pathToPythonFile) == 0
%     insert(py.sys.path, int32(0), pathToPythonFile);
% end

% pathToPythonFile
% Note, it is picking up Python2 as my default Python
% Will it matter?

% pyOut = py.rtn.computeGRF();
% myText = string(pyOut{2});
% disp(myText)

commandStr = 'python3 /home/peter/Desktop/Chrono/chrono/template_project/matlab-with-python/rtn.py';
system(commandStr)

% This is how to define functions in Matlab
% Define one for each function in the Python code
%function [F1, F2, F3] = predict_GRF(x)
% Fill in data here
%end

% Define a bunch of variables to compute each gradient we want 
%function [m, s] = dX_dN(x)
% Fill in data here
%end

