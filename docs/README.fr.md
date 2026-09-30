# ACM — Laboratoire public d'apprentissage adaptatif

**Version 0.3 expérimentale.** Nous étudions deux capacités distinctes : enregistrer une information dès sa première observation et réutiliser des liens entre connaissances déjà disponibles.

## Capacités actuelles

1. **v0.1** : mémoire associative de vecteurs numériques (similarité cosinus).
2. **v0.2** : interface d'encodage d'images avec OpenCLIP préentraîné et gelé (pas encore de résultats mesurés sur de vraies photos).
3. **v0.3** : entrée de faits textuels en français/anglais **dans une grammaire limitée**, association sujet–relation–objet, sources conservées, contradictions explicites, inférence sur la relation de catégorie `is_a` ; connexion expérimentale des informations textuelles aux images par étiquette identique.

Un exemple :

```text
Zentra est une marque automobile.
Une marque automobile est une organisation.
=> Est-ce que Zentra est une organisation ?
=> supported : chemin des deux relations conservées.
```

Ce résultat vient d'une règle de transitivité déjà codée et des deux phrases fournies. **Il ne démontre ni compréhension générale du français ni acquisition autonome de concepts.**

## Tester

Avec Python 3.10+ :

```bash
python -m unittest discover -s tests -v
python -m examples.text_one_shot
```

Pour la vision, voir [docs/VISION.md](VISION.md). Pour les règles et limites du texte, voir [docs/TEXT.md](TEXT.md).

## Principes de recherche

- Séparer faits mémorisés, inférences et vérification de vérité.
- Conserver les sources et contradictions ; signaler l'absence de preuve.
- Comparer le système aux méthodes classiques de graphes et de recherche.
- Publier aussi les échecs, les ressources consommées et les limites.

Le prochain véritable défi est de créer des concepts nouveaux, pas seulement de stocker ou retrouver des relations explicites.
