# Cartpole

In het `cartpole.py` bestand staat een uitgewerkt reinforcement learning algoritme die de [cartpole-v0](https://gym.openai.com/envs/CartPole-v0/) omgeving van gym kan oplossen. Deze uitwerking is bedoeld als voorbeeld om je een idee te geven hoe een wat meer geadvanceerd RL algoritme er uit kan komen te zien om een complex(er) probleem op te kunnen lossen.

Je kunt het algoritme trainen met het commando:

```python cartpole.py```

Als het trainen klaar is, of als je het trainen korttijdig afbreekt wordt het getrainde model als .h5 bestand opgeslagen. Je kunt de kwaliteit van een getraind model met het volgende commando uitvoeren:

```python cartpole.py --test model.h5```

Als je wilt, kun je de parameters of de implementatie aanpassen om te zien of je de kwaliteit of efficiëntie van het algoritme kan verbeteren. 
