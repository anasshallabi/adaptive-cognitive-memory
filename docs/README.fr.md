# ACM — Laboratoire public d'apprentissage adaptatif (v0.7 — preuves révisables)

## Vision

Étudier une IA capable de mémoriser immédiatement une information, de la relier à ses connaissances et de la réutiliser sans devoir réentraîner un gigantesque modèle à chaque observation. Nous séparons l'apprentissage d'un *nouveau fait* de l'acquisition d'un *nouveau concept*.

## État du code

- **v0.1** — mémoire de vecteurs et similarité.
- **v0.2** — perception d'images avec OpenCLIP préentraîné, tests visuels réels encore à mener.
- **v0.3** — phrases françaises/anglaises dans une grammaire limitée ; mémorisation de faits sourcés, inférences `is_a`, contradictions visibles.
- **v0.4** — **extracteur de texte optionnel via Ollama local**, pour essayer des formulations qui dépassent la grammaire codée. Le modèle doit être préentraîné ; ACM conserve les faits extraits sans modifier les poids de ce modèle. Banc d'essai synthétique de 20 phrases, avec validation et test séparés.

- **v0.5 (évaluation)** — nouveau corpus synthétique de **40 phrases**, étiquetées avant exécution de Qwen3 : 12 en développement, 28 en test. Il distingue affirmations directes, négations, hypothèses, propos rapportés et faits historiques. Les résultats Qwen3 restent **à mesurer**. [Protocole v0.5](EVALUATION_V05.md).

## Premiers résultats réels de v0.5

Sur le PC d'Anass, Qwen3 14B + règles a réussi **12/12 phrases
de validation**, mais seulement **13/28 phrases de test (46,4 %)**.
Parmi les 10 phrases qui exigeaient une abstention, le système n'en
a rejeté correctement que 4. Des règles trop larges ont parfois
transformé des formulations ambiguës en faits stockables.

L'analyse et la liste des erreurs se trouvent dans
[le rapport v0.5](RESULTS_V05_2026-09-30.md).

Une première expérimentation **v0.6** propose un
[filtre conservateur facultatif](CONSERVATIVE_GATE.md). Ce filtre
heuristique peut refuser les phrases à risque avant leur mémorisation,
mais ses performances sur des phrases vraiment nouvelles **restent à tester**.

## Résultat du filtre et nouvelle mémoire v0.7

Le mode conservateur a obtenu **21/28 (75 %)** contre **13/28
(46,4 %)** pour le moteur hybride initial, sur **les mêmes phrases
déjà étudiées**. Les dix abstentions attendues ont été respectées.
Ce résultat est un test de régression *après analyse des erreurs* :
il ne mesure pas une généralisation indépendante. Voir le
[rapport de v0.5 et du filtre](RESULTS_V05_2026-09-30.md).

La mémoire v0.7 introduit un **registre de preuves révisables**.
Une nouvelle affirmation reste `candidate` jusqu'à une décision
explicite d'un examinateur. Le système conserve la source, les
décisions, leurs motifs et les rétractations, et exclut les
affirmations contestées de son graphe d'inférence approuvé.
Ce prototype ne vérifie pas automatiquement la vérité :
[documentation complète](EVIDENCE_LEDGER.md).

Le test de démonstration ne nécessite ni Ollama ni GPU :

```powershell
python -m examples.evidence_review
```

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
