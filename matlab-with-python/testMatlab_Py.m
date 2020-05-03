
pyversion
pathToSpeech = fileparts('/home/peter/Desktop/Chrono/chrono/template_project/matlab-with-python/helloWord.py');
if count(py.sys.path,pathToSpeech) == 0
    insert(py.sys.path,int32(0),pathToSpeech);
end

pathToSpeech

pyOut = py.helloWorld.myFunction();
myText = string(pyOut{2});

% successFlag = logical(pyOut{2});
disp(myText)

% Pass data to a function