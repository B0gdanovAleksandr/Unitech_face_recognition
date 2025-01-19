echo "Установка приложения ..."

#~ sudo apt install gcc-11 git make build-essential libffi-dev libssl-dev zlib1g-dev libbz2-dev liblzma-dev libsqlite3-dev libreadline-dev tk-dev

if ! pyenv root; then
	echo "Установка pyenv..."
	git clone https://github.com/pyenv/pyenv.git ~/.pyenv
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init -)"' >> ~/.bashrc
	source ~/.bashrc
fi

echo "Установка python 3.7.17 ..."
pyenv install -s 3.7.17
pyenv local 3.7.17

#~ echo "Создание виртуального окружения ..."
#~ python -m venv .venv
#~ source .venv/bin/activate

echo "Установка необходимых модулей ..."
python -m pip install --upgrade pip
python -m pip install -r dotnet_requirements.txt
python -m pip uninstall -y opencv-python-headless
python -m pip uninstall -y opencv-python
python -m pip install opencv-contrib-python
python -m pip install opencv-python

#~ deactivate

if ! dotnet --version; then
	echo "Установка dotnet..."
	wget https://dot.net/v1/dotnet-install.sh
	source dotnet-install.sh
	echo 'export PATH="$PATH:$HOME/.dotnet/"' >> ~/.bashrc
	export PATH="$PATH:$HOME/.dotnet/"
fi

cd dotnet/
dotnet add package Python.Runtime.Linux

echo "Установка завершена."

echo "Для запуска выполните:"
echo "$PWD/run.sh"
