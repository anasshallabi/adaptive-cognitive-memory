# ACM — Laboratoire public d'apprentissage adaptatif (v0.5 — protocole d'évaluation)

## Vision

Étudier une IA capable de mémoriser immédiatement une information, de la relier à ses connaissances et de la réutiliser sans devoir réentraîner un gigantesque modèle à chaque observation. Nous séparons l'apprentissage d'un *nouveau fait* de l'acquisition d'un *nouveau concept*.

## État du code

- **v0.1** — mémoire de vecteurs et similarité.
- **v0.2** — perception d'images avec OpenCLIP préentraîné, tests visuels réels encore à mener.
- **v0.3** — phrases françaises/anglaises dans une grammaire limitée ; mémorisation de faits sourcés, inférences `is_a`, contradictions visibles.
- **v0.4** — **extracteur de texte optionnel via Ollama local**, pour essayer des formulations qui dépassent la grammaire codée. Le modèle doit être préentraîné ; ACM conserve les faits extraits sans modifier les poids de ce modèle. Banc d'essai synthétique de 20 phrases, avec validation et test séparés.

- **v0.5 (évaluation)** — nouveau corpus synthétique de **40 phrases**, étiquetées avant exécution de Qwen3 : 12 en développement, 28 en test. Il distingue affirmations directes, négations, hypothèses, propos rapportés et faits historiques. Les résultats Qwen3 restent **à mesurer**. [Protocole v0.5](EVALUATION_V05.md).

## Commandes

```powershell
python -m unittest discover -s tests -v
python -m examples.compare_text --extractor rules --split test
```

Pour comparer avec un modèle Ollama *installé localement* :

```powershell
python -m examples.compare_text --extractor hybrid --model gemma3:4b --split test
```

Sans Ollama, le mode `rules` est gratuit, sans installation Python supplémentaire. Pour le mode `hybrid`, le calcul du modèle tourne sur ta machine, et consomme des ressources locales, sans nécessiter de crédit API externe.

Premiers résultats locaux : avec Qwen3 14B, le moteur hybride a obtenu **4 réponses exactes sur 12**, soit le même résultat que les règles seules, avec davantage de calcul. [Rapport détaillé et résultat négatif](RESULTS_V04_2026-09-30.md).

La phrase « Zentra est considéré comme une marque » a suscité une abstention. Nous étudions la différence entre **fait direct**, **fait rapporté**, **hypothèse** et **affirmation datée**, sans changer les étiquettes historiques du test. [Note de recherche sur ces nuances](MODALITY.md).

Pour sonder deux phrases sans relancer tout le benchmark :

```powershell
python -m examples.probe_text --model qwen3:14b --case direct --case attributed
```

Le prompt alternatif a extrait une affirmation directe en français et s'est abstenu sur une attribution et une possibilité, **sur trois exemples de diagnostic seulement**. Ces résultats sont archivés dans [notre journal](RESULTS_V04_2026-09-30.md) ; ils ne prouvent pas une généralisation.

Consulte aussi [l'expérience v0.4](EXTRACTION.md), [le moteur textuel v0.3](TEXT.md), [le module visuel](VISION.md) et [la méthode d'évaluation](EVALUATION.md).

**Limite essentielle :** reconnaître une paraphrase avec un LLM déjà entraîné n'est pas découvrir un concept à partir de zéro. Nous n'avons pas encore validé de nouvelle architecture cognitive ; la recherche continue.
