# racebaan
code om auto's op de racebaan te laten rijden

#### Code voor beeldherkenning (https://github.com/anujshah1003/Real_time_Object_detection_TF)
* `git clone https://github.com/anujshah1003/Real_time_Object_detection_TF.git`
* Vervang volgens het python-bestand object_detection_webcam.py door het bestand wat in deze repository staat (hier staan ook de code in om de auto's aan te sturen)

#### Code voor auto's laten rijden (overdrive: https://github.com/xerodotc/overdrive-python)
* `git clone https://github.com/xerodotc/overdrive-python.git`
* Verplaats de inhoud van deze map (onder andere overdrive.py) naar dezelfde folder als waarin je net object_detection_webcam.py hebt vervangen

#### Installeer de packages uit de req_cars.txt file in een conda environment:
* `conda create --name overdrive python=3.7.3 --file req_cars_updated.txt`

#### Activeer deze environment met het volgende commando:
* `conda activate overdrive`

#### En installeer nu de opencv library via een andere channel
* `conda install -c conda-forge opencv=4.1.0`

#### Installeer dan met behulp van pip de bluepy package, de --user optie is hierbij nodig
* `python3.7 -m pip install bluepy --user`
* Dit script wordt zo gebruikt omdat pip anders in de war kan raken met andere versies van python die op de laptop staan


#### Open vervolgens het script object_detection_webcam.py in een editor en bekijk regel 103, hier zien we:
* `cap = cv2.VideoCapture(0)`
* Selecteer hier 0 om de camera in je laptop te gebruiken, en selecteer 1 om een externe camera aan te spreken

#### Vervolgens kunnen we het script starten:
* `cd /real-time-Object-detection_TF-master/object_recognition_master/object_recognition_detection/
*  python object_detection_webcam.py [beginsnelheid_auto]
*  beginsnelheid_auto is hier een integerwaarde
