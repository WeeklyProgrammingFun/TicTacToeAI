Write-Output "Create python virtual environment"
py -3.8 -m venv venv
./venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Output "Run the program"
Set-Location -Path Network -PassThru
python .\Net.py
tensorboard dev upload --logdir=runs --name "WPF-TicTacToe" --one_shot

Write-Output "Cleanup"
Set-Location -Path .. -PassThru
deactivate