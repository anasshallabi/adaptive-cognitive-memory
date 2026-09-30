# ACM — Documentation française (v0.2)

## Notre objectif
Explorer une IA qui retient une information après une seule observation et la relie à ses connaissances, sans réentraîner tout son réseau neuronal.

## Ce qui est implémenté
- Mémoire associative v0.1 : vecteurs numériques, étiquettes et relations explicites.
- Nouveau module v0.2 : conversion de **vraies photos** en représentations numériques via **OpenCLIP préentraîné et gelé** ; association d'une image et d'un nom ; tentative de reconnaissance de nouvelles images.
- Évaluateur CSV contrôlant certains risques de fuite de données entre voitures / séances de prise de vue.
- Tests sans téléchargement de modèle.

**Aucun score sur de vraies photographies n'a été mesuré.** OpenCLIP possède déjà des connaissances acquises au préentraînement : une association en un exemple ne signifie pas une compréhension nouvelle de zéro.

## Installation et expérience
[Guide complet Windows, CUDA et commandes](VISION.md) ; [architecture](ARCHITECTURE.md) ; [évaluation](EVALUATION.md).

Après installation PyTorch/OpenCLIP, avec trois photos locales autorisées :

```powershell
python -m examples.vision_one_shot --support data/toyota_a.jpg --label Toyota --query data/toyota_b.jpg data/honda.jpg --device auto
```

La similarité calculée n'est pas une probabilité ; le seuil d'exemple n'est pas calibré. Nous devons vérifier séparément la reconnaissance d'un logo visible et la généralisation à une autre carrosserie.
