Last reviewed: 2025-09-27

# bl-counterfactuals

Zweck: Generierung von Gegenfaktischen (Counterfactuals) zur Ableitung konkreter Maßnahmen, die das Churn‑Risiko senken können.

## Architektur-Überblick
- Datenzugriff/DAL und zentrale Pfade: via Projekt‑Konfiguration
- Input: Modellscore/Features aus `bl-churn`
- Output/Artefakte: `bl-churn/dynamic_system_outputs/outbox/counterfactuals/experiment_<id>/cf_*.json`
- Kostenmodell: `config/shared/config/cf_cost_policy.json`

## Wichtige Artefakte (Beispiele)
- `cf_business_metrics.json`: kundenbezogene Nutzenrechnung (p_old/p_new, Reduktion, Kosten, ROI)
- `cf_feature_recommendations.json`: Top‑Empfehlungen je Kunde (Delta, Kosten, Effizienz)
- `cf_aggregate.json`: aggregierte Feature‑Deltas
- `cf_cost_analysis.json`: Kosten je Roh‑Feature und abgeleitete Features

Ausführliche Befunde/Verbesserungsvorschläge siehe `nextSteps.md`.

## Runbook
Siehe `RUNBOOK.md` für Setup/Abhängigkeiten und Laufhinweise.
