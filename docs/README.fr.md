# ACM — Documentation française

## Vision

Étudier une IA qui associe une information nouvelle dès une observation, la relie à ses connaissances antérieures et la réutilise sans réentraînement global. L'exemple conducteur est la découverte d'une marque automobile inconnue.

## État actuel (v0.1)

**Il s'agit seulement d'une base expérimentale.** Le programme mémorise des vecteurs numériques fournis par l'utilisateur, associe des étiquettes à des concepts explicites, puis calcule une similarité cosinus. Il **ne lit pas encore d'images** et ne construit pas de concepts de façon autonome. La démonstration utilise des vecteurs artificiels. Aucun gain d'efficacité ni capacité comparable à celle d'un humain n'a été prouvé.

## Lancer

Installer Python 3.10+ puis :

```bash
python -m unittest discover -s tests -v
python -m examples.one_shot
```

## Objectifs expérimentaux

1. Retenir une association après un seul exemple, sans modifier les poids d'un encodeur.
2. Vérifier sa généralisation sur d'autres véhicules/images non vus.
3. Quantifier les confusions entre marques, le rejet d'inconnus et l'oubli après ajout de connaissances.
4. Comparer qualité et coûts de calcul à une méthode de référence, sur le même matériel.

La distinction essentielle : mémoriser un nom n'est pas comprendre un concept. Voir [protocole](EVALUATION.md), [architecture](ARCHITECTURE.md) et [questions scientifiques](RESEARCH.md).
