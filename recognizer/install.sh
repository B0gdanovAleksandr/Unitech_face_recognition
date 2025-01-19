
destination_path=$1

if [ -z "$destination_path" ]; then
	destination_path="$HOME/.local/recognizer"
fi

echo "Копирование в $destination_path ..."

mkdir -p $destination_path/unitech
mkdir -p $destination_path/tools

files="run.sh recognizer.py setup.sh dotnet_requirements.txt unitech/detector_Pluzhnikov.py unitech/Reznik.py tools/view.py unitech/detector.py unitech/streams.py"

for path in $files; do
	echo $path
	cp -r $path $destination_path/$path
done

cp -r dotnet $destination_path
cp -r unitech/AntiSpoof $destination_path/unitech
cp -r unitech/saved_models $destination_path/unitech


cd $destination_path
source setup.sh