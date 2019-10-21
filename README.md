# racebaan
code om auto's op de racebaan te laten rijden

Code voor beeldherkenning (https://github.com/anujshah1003/Real_time_Object_detection_TF)
git clone https://github.com/anujshah1003/Real_time_Object_detection_TF.git
Vervang volgens het python-bestand object_detection_webcam.py door het bestand wat in deze repository staat (hier staan ook de code in om de auto's aan te sturen)

Code voor auto's laten rijden (overdrive: https://github.com/xerodotc/overdrive-python)
git clone https://github.com/xerodotc/overdrive-python.git

Installeer de volgende packages in een conda environment:
conda create --name overdrive python=3.7.3
conda install tensorflow==1.14
conda install spyder
conda install pillow
conda install matplotlib
conda install opencv
pip install bluepy

conda activate overdrive 
cd Documenten/demo/real-time-Object-detection_TF-master/object_recognition_master/object_recognition_detection/
python object_detection_webcam.py
