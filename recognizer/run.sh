script_path=$(dirname ${BASH_SOURCE[0]})
cd $script_path

#~ python recognizer.py
cd dotnet/
export LD_LIBRARY_PATH=~/.pyenv/versions/3.7.17/lib/
dotnet run