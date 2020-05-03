

pyversion
pathToPythonFile = fileparts('/home/peter/Desktop/Chrono/chrono/template_project/matlab-with-python');
if count(py.sys.path,pathToSpeech) == 0
    insert(py.sys.path, int32(0), pathToSpeech);
end

pathToPythonFile
% Note, it is picking up Python2 as my default Python
% Will it matter?

% Example of passing data to/from Python
pyOut = py.helloWorld.myFunction();
myText = string(pyOut{4});
disp(myText)


% This is how to define functions in Matlab
% Define one for each function in the Python code
function [m, s] = predict_GRF(x)
    % Fill in data here
end

% Define a bunch of variables to compute each gradient we want 
function [m, s] = dX_dN(x)
   % Fill in data here
end

