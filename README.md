# Innovatiedag Machine Learning

Om met deze repo aan de slag te gaan heb je de volgende dingen nodig:

- Voor opdrachten 1 & 2 heb je Docker nodig. 
- Voor het extra materiaal heb je Python 3.5 of 3.6 nodig (ik raad zelf 3.6 aan).

## Docker - voorbereiding opdrachten 1 & 2

Docker kun je installeren via https://docs.docker.com/install/.

Na het clonen van deze repo kan de Dockerfile in dezelfde directory worden gebouwd met het commando:

```docker build -f Dockerfile -t jupyter-ml .```

Als de docker container klaar is, kun je deze uitvoeren met het volgende commando:

**Windows**

```docker run --rm -p 8888:8888 -v ${PWD}:/home/jovyan/work jupyter-ml:latest```

**Linux en MacOS**

```docker run --rm -p 8888:8888 -v $PWD:/home/jovyan/work jupyter-ml:latest```

Dit commando zorgt ervoor dat de huidige directory toegankelijk is voor de docker omgeving zodat de .ipynb bestanden daarin geopend kunnen worden, en gemaakte wijzigingen ook weer op jouw lokale machine terecht komen. In de terminal wordt vervolgens een url in de volgende vorm getoond: *"http://127.0.0.1:8888/?token=YOUR_TOKEN"*. Open deze url in jouw favoriete webbrowser. Als het goed is krijg je nu het Jupyter notebook dashboard te zien.

## Extra materiaal - voorbereiding

Om het extra materiaal te kunnen gebruiken heb je een installatie van Python 3.5 of 3.6 nodig (ik raad zelf 3.6 aan). Via het `pip` commando kun je de benodigde python modules installeren die in de `requirements.txt` file staan opgegeven:

```pip install -r requirements.txt```

Als je meerdere Python versies hebt geïnstalleerd, gebruik dan het `pip3` commando.

Zie de bijbehorende readme bestanden voor overige informatie over hoe je het materiaal kan draaien.
