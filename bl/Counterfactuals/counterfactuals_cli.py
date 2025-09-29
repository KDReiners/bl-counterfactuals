#!/usr/bin/env python3
"""
Counterfactuals CLI (ohne Notebook)
===================================

Ziel
- Erzeuge Counterfactuals rein aus CUSTOMER_DETAILS (engineered Features),
  ohne Rohdaten nachzuladen.
- Policy in Roh-Feature-Namen wird via feature_mapping.json auf engineered
  Features gemappt (beim Laden der Policy).
- Reports zusätzlich als Roh-Feature-Namen aggregiert ausgeben.

Ausgaben
- artifacts/<experiment_id>/counterfactuals/
  - cf_individual.json
  - cf_aggregate.json
  - cf_individual_raw.json
  - cf_aggregate_raw.json

Aufruf
  python "Trash potential/counterfactuals_cli.py" --experiment-id 1 --sample 0.2
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.paths_config import ProjectPaths
from bl.json_database.sql_query_interface import SQLQueryInterface
from bl.json_database.leakage_guard import load_cf_cost_policy
from bl.json_database.churn_json_database import ChurnJSONDatabase


# =============================
# Utility: Feature selection
# =============================

META_COLS = {
    "experiment_id",
    "Kunde",
    "I_ALIVE",
    "I_Alive",
    "Churn_Wahrscheinlichkeit",
    "Letzte_Timebase",
    "yyyymm",
    "source",
    "analysis_date",
    "model_type",
    "experiment_name",
}

FORBIDDEN_PREFIXES = (
    "Predicted_",
    "Threshold_",
    "p_survival_",
    "p_event_",
    "rmst_",
)

FORBIDDEN_EXACT = {
    "Churn_Wahrscheinlichkeit",
    "P_Event_6m",
    "P_Event_12m",
    "P_Event_24m",
    "source",
    "cox_analysis_type",
    "customer_status",
    "analysis_date",
    "model_performance",
    "risk_category",
    "experiment_name",
    "model_type",
    "experiment_date",
    "Letzte_Timebase",
    "t_end",
    "expected_lifetime_unconditional",
    "Threshold_Standard_0.5",
    "priority_score",
    "expected_lifetime_months",
    "Threshold_Optimal",
    "t_start",
    "rmst_24m",
    "Threshold_Elbow",
    "p_survival_24m",
    "rmst_12m",
    "feature_count",
    "duration",
    "event",
    "survival_months",
    "event_occurred",
    "c_index",
}


def load_feature_mapping() -> Dict[str, List[str]]:
    """Roh→engineered Mapping laden (falls vorhanden)."""
    path = ProjectPaths.feature_mapping_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def engineered_feature_set_from_mapping(mapping: Dict[str, List[str]]) -> List[str]:
    feats: List[str] = []
    for _, lst in mapping.items():
        if isinstance(lst, list):
            feats.extend([x for x in lst if isinstance(x, str)])
    return sorted(set(feats))


def choose_policy_features(df: pd.DataFrame, policy: Policy) -> List[str]:
    """Nur Features aus cf_cost_policy.json verwenden. Keine Fallbacks.
    Policy-Keys können Raw-Basisnamen sein (z. B. I_UHD). In dem Fall werden
    genau die df-Spalten zugelassen, die entweder exakt so heißen oder mit
    "<RAW>_" beginnen. Output-/Meta-Spalten werden strikt ausgeschlossen.
    """
    if not isinstance(policy.features, dict) or not policy.features:
        raise RuntimeError("Policy enthält keine Features (cf_cost_policy.json).")
    raw_keys = [k for k in policy.features.keys() if isinstance(k, str)]
    selected: List[str] = []
    for c in df.columns:
        if c in FORBIDDEN_EXACT or any(str(c).startswith(p) for p in FORBIDDEN_PREFIXES) or c in META_COLS:
            continue
        for raw in raw_keys:
            if c == raw or str(c).startswith(f"{raw}_") or (raw == "N_DIGITALIZATIONRATE" and str(c).startswith("N_DIGITALIZATIONRATE")):
                selected.append(c)
                break
    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError("Keine geeigneten Policy-Features in CUSTOMER_DETAILS gefunden.")
    return selected


# =============================
# Policy / Distance
# =============================

@dataclass
class Policy:
    default_step: float
    features: Dict[str, Dict[str, Any]]


def load_policy() -> Policy:
    p = load_cf_cost_policy()  # bereits mit Mapping expandiert
    return Policy(
        default_step=float(p.get("default_step", 0.1)),
        features=p.get("features", {}) if isinstance(p.get("features", {}), dict) else {},
    )


def project_value(name: str, val: float, policy: Policy) -> float:
    pol = policy.features.get(name, {})
    step = float(pol.get("step", policy.default_step))
    lo = pol.get("min")
    hi = pol.get("max")
    t = str(pol.get("type", "")).lower()
    if t == "binary":
        return float(1.0 if val >= 0.5 else 0.0)
    if t == "integer":
        val = float(int(round(val)))
    else:
        val = round(val / step) * step
    if lo is not None:
        val = max(val, float(lo))
    if hi is not None:
        val = min(val, float(hi))
    return float(val)


def weighted_l2(a: np.ndarray, b: np.ndarray, names: List[str], policy: Policy) -> float:
    weights = np.array([float(policy.features.get(n, {}).get("weight", 1.0)) for n in names], dtype=float)
    diff = (a - b)
    return float(np.sqrt(((weights * diff) ** 2).sum()))


# =============================
# Model / Threshold
# =============================

def build_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve

    class Wrapper:
        def fit(self, X, y):
            base = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            # sklearn 1.5+: estimator, ältere: base_estimator
            try:
                self.clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
            except TypeError:
                self.clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=3)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.clf.fit(X_train, y_train)
            probs = self.clf.predict_proba(X_test)[:, 1]
            p, r, th = precision_recall_curve(y_test, probs)
            f1 = (2 * p * r) / np.maximum(p + r, 1e-9)
            if th.size == 0 or np.all(np.isnan(f1)):
                self.tau = 0.5
            else:
                idx = int(np.nanargmax(f1))
                self.tau = float(th[max(0, idx - 1)])
            return self

        def predict_proba(self, X):
            return self.clf.predict_proba(X)

        def threshold(self):
            return float(getattr(self, "tau", 0.5))

    return Wrapper()


def build_surrogate_prob_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    class Wrapper:
        def fit(self, X, y_prob):
            # Regressor, der die vorhandene RF-Probability approximiert
            self.reg = RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_prob, test_size=0.2, random_state=42
            )
            self.reg.fit(X_train, y_train)
            try:
                y_hat = self.reg.predict(X_test)
                _ = r2_score(y_test, y_hat)
            except Exception:
                pass
            return self

        def predict_proba(self, X):
            p = self.reg.predict(X)
            p = np.clip(p, 0.0, 1.0)
            return np.stack([1.0 - p, p], axis=1)

        def threshold(self):
            return 0.5

    return Wrapper()


# =============================
# CF Search (Greedy coordinate descent)
# =============================

def find_counterfactual(x0_raw: np.ndarray,
                        names: List[str],
                        model,
                        scaler,
                        policy: Policy,
                        max_iter: int = 200,
                        p_target: Optional[float] = None,
                        tau: Optional[float] = None) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    # Suche im Roh-Feature-Raum; Bewertung im Modellraum (via scaler)
    z = x0_raw.copy()
    best_dist = weighted_l2(z, x0_raw, names, policy)
    changes: Dict[int, float] = {}
    for _ in range(int(max_iter)):
        p = float(model.predict_proba(scaler.transform(z.reshape(1, -1)))[0, 1])
        # Ziel: bevorzugt p_target (relative Reduktion), sonst legacy tau
        if p_target is not None:
            if p <= float(p_target):
                break
        elif tau is not None:
            if p < float(tau):
                break
        improved = False
        for j, name in enumerate(names):
            pol = policy.features.get(name, {})
            step = float(pol.get("step", policy.default_step))
            for direction in (-1.0, 1.0):
                cand = z.copy()
                new_val = project_value(name, cand[j] + direction * step, policy)
                if float(new_val) == float(cand[j]):
                    continue  # keine effektive Änderung
                cand[j] = float(new_val)
                p_c = float(model.predict_proba(scaler.transform(cand.reshape(1, -1)))[0, 1])
                d_c = weighted_l2(cand, x0_raw, names, policy)
                if p_c < p:
                    z = cand
                    best_dist = d_c
                    changes[j] = float(z[j] - x0_raw[j])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    # changes in reportable Form
    top_changes = [
        {"feature": names[j], "delta": float(d), "abs_delta": float(abs(d))}
        for j, d in changes.items()
        if abs(d) > 0
    ]
    top_changes.sort(key=lambda r: r["abs_delta"], reverse=True)
    return z, best_dist, top_changes


# =============================
# Raw-name Reporter (inline)
# =============================

def invert_mapping(mapping: Dict[str, List[str]]) -> Dict[str, str]:
    inv = {}
    for raw, lst in mapping.items():
        for eng in lst:
            inv[eng] = raw
    return inv


def build_raw_reports(individual: List[Dict[str, Any]], aggregate: List[Dict[str, Any]], mapping: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    inv = invert_mapping(mapping)
    # individual
    ind_out = []
    for rec in individual:
        raw_changes: Dict[str, Dict[str, float]] = {}
        for ch in rec.get("top_changes", []):
            f = ch.get("feature")
            raw = inv.get(f, f)
            ent = raw_changes.setdefault(raw, {"delta": 0.0, "abs_delta": 0.0})
            ent["delta"] += float(ch.get("delta", 0.0))
            ent["abs_delta"] += float(ch.get("abs_delta", 0.0))
        ind_out.append({
            **{k: v for k, v in rec.items() if k != "top_changes"},
            "top_changes_raw": [
                {"feature": k, "delta": v["delta"], "abs_delta": v["abs_delta"]}
                for k, v in raw_changes.items()
            ]
        })
    # aggregate
    agg_by_raw: Dict[str, Dict[str, float]] = {}
    for a in aggregate:
        raw = inv.get(a.get("feature"), a.get("feature"))
        ent = agg_by_raw.setdefault(raw, {"count": 0.0, "median_abs_delta": 0.0})
        ent["count"] += float(a.get("count", 0))
        ent["median_abs_delta"] = max(ent["median_abs_delta"], float(a.get("median_abs_delta", 0.0)))
    agg_out = [
        {"feature": k, "count": int(v["count"]), "median_abs_delta": float(v["median_abs_delta"])}
        for k, v in agg_by_raw.items()
    ]
    return ind_out, agg_out


# =============================
# Business Metrics Creation
# =============================

def _create_business_metrics(
    individual: List[Dict[str, Any]], 
    aggregate: List[Dict[str, Any]], 
    policy, 
    raw_to_eng: Dict[str, List[str]], 
    experiment_id: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Erstellt Business-Metriken für CF-Analyse:
    - cf_business_metrics: ROI, Kosten pro Kunde
    - cf_feature_recommendations: aufgelöste Feature-Empfehlungen mit Kosten
    - cf_cost_analysis: Feature-Kosten-Übersicht aus Policy
    """
    
    # 1. cf_cost_analysis: Policy-Kosten pro Feature
    cost_analysis = []
    for raw_feature, config in policy.features.items():
        # Handle both dict and object configs
        if isinstance(config, dict):
            weight = config.get('weight', 1.0)
            feature_type = config.get('type', 'unknown')
            step_size = config.get('step', 1)
            min_value = config.get('min', None)
            max_value = config.get('max', None)
        else:
            weight = getattr(config, 'weight', 1.0)
            feature_type = getattr(config, 'type', 'unknown')
            step_size = getattr(config, 'step', 1)
            min_value = getattr(config, 'min', None)
            max_value = getattr(config, 'max', None)
            
        cost_analysis.append({
            "raw_feature": raw_feature,
            "weight": weight,
            "feature_type": feature_type,
            "step_size": step_size,
            "min_value": min_value,
            "max_value": max_value,
            "engineered_features": raw_to_eng.get(raw_feature, []),
            "engineered_count": len(raw_to_eng.get(raw_feature, [])),
            "experiment_id": experiment_id
        })
    
    # 2. cf_feature_recommendations: Aufgelöste top_changes pro Kunde mit Kosten
    feature_recommendations = []
    for customer in individual:
        if customer.get("no_cf_found"):
            continue
            
        customer_id = customer.get("Kunde")
        p_old = customer.get("p_old", 0.0)
        p_new = customer.get("p_new", 0.0)
        total_cost = customer.get("l2_weighted", 0.0)  # L2-gewichtete Kosten
        
        # Analysiere top_changes
        top_changes = customer.get("top_changes", [])
        for i, change in enumerate(top_changes):
            feature_name = change.get("feature", "")
            delta = change.get("delta", 0.0)
            abs_delta = change.get("abs_delta", 0.0)
            
            # Finde zugehörigen raw_feature und Kosten
            raw_feature = None
            feature_cost = 0.0
            for raw, eng_list in raw_to_eng.items():
                if feature_name in eng_list:
                    raw_feature = raw
                    raw_config = policy.features.get(raw, {'weight': 1.0})
                    weight = raw_config.get('weight', 1.0) if isinstance(raw_config, dict) else getattr(raw_config, 'weight', 1.0)
                    feature_cost = weight * abs_delta
                    break
            
            feature_recommendations.append({
                "customer_id": customer_id,
                "recommendation_rank": i + 1,
                "feature_name": feature_name,
                "raw_feature": raw_feature or "unknown",
                "delta": delta,
                "abs_delta": abs_delta,
                "feature_cost": feature_cost,
                "p_old": p_old,
                "p_new": p_new,
                "total_customer_cost": total_cost,
                "relative_reduction": (p_old - p_new) / p_old if p_old > 0 else 0.0,
                "cost_per_reduction": feature_cost / ((p_old - p_new) / p_old) if p_old > p_new else float('inf'),
                "experiment_id": experiment_id
            })
    
    # 3. cf_business_metrics: Customer-ROI-Metriken
    business_metrics = []
    
    # Annahme: Durchschnittlicher Customer Lifetime Value und Churn-Kosten
    # Diese sollten idealerweise aus der Konfiguration kommen
    avg_customer_value = 1000.0  # EUR - sollte konfigurierbar sein
    churn_cost_factor = 0.2  # 20% des CLV als Churn-Kosten
    
    for customer in individual:
        if customer.get("no_cf_found"):
            continue
            
        customer_id = customer.get("Kunde")
        p_old = customer.get("p_old", 0.0)
        p_new = customer.get("p_new", 0.0)
        total_cost = customer.get("l2_weighted", 0.0)
        
        # Business-Metriken berechnen
        churn_risk_reduction = p_old - p_new
        relative_reduction = churn_risk_reduction / p_old if p_old > 0 else 0.0
        potential_savings = churn_risk_reduction * avg_customer_value * churn_cost_factor
        roi = (potential_savings - total_cost) / total_cost if total_cost > 0 else float('inf')
        
        # Empfehlungsstatus
        recommendation = "highly_recommended" if roi > 3.0 else \
                        "recommended" if roi > 1.0 else \
                        "consider" if roi > 0.0 else \
                        "not_recommended"
        
        business_metrics.append({
            "customer_id": customer_id,
            "p_old": p_old,
            "p_new": p_new,
            "churn_risk_reduction": churn_risk_reduction,
            "relative_reduction": relative_reduction,
            "implementation_cost": total_cost,
            "potential_savings": potential_savings,
            "net_benefit": potential_savings - total_cost,
            "roi": roi if roi != float('inf') else 999.99,
            "recommendation": recommendation,
            "avg_customer_value": avg_customer_value,
            "churn_cost_factor": churn_cost_factor,
            "experiment_id": experiment_id
        })
    
    return {
        "cf_cost_analysis": cost_analysis,
        "cf_feature_recommendations": feature_recommendations,
        "cf_business_metrics": business_metrics
    }


# =============================
# Main
# =============================

def run(experiment_id: int, sample: float = 0.2, limit: int = 0) -> bool:
    iface = SQLQueryInterface()
    sql = f"""
    SELECT *
    FROM customer_churn_details
    WHERE experiment_id = {int(experiment_id)}
    {"AND random() < " + str(float(sample)) if sample and 0 < sample < 1 else ""}
    """
    df = iface.execute_query(sql, output_format="pandas")
    if isinstance(df, str):
        # Fallback 1: direkt mit DuckDB ausführen und DataFrame bauen
        try:
            records = iface._execute_with_duckdb(sql)
            df = pd.DataFrame(records)
        except Exception:
            # Fallback 2: JSON-DB direkt lesen
            try:
                db = ChurnJSONDatabase()
                records = db.data.get("tables", {}).get("customer_churn_details", {}).get("records", []) or []
                df = pd.DataFrame(records)
                df = df[df.get("experiment_id") == int(experiment_id)] if "experiment_id" in df.columns else df
            except Exception:
                raise RuntimeError(f"SQL-Interface Fehler: {df}")
    if df is None or getattr(df, 'empty', True):
        raise RuntimeError("Keine Daten in customer_churn_details für dieses Experiment gefunden.")

    # Label
    y = df.get("I_ALIVE") if "I_ALIVE" in df.columns else df.get("I_Alive")
    if y is None:
        raise RuntimeError("Spalte I_ALIVE/I_Alive nicht gefunden.")
    y = (~df["I_ALIVE"].astype(str).str.lower().isin(["true", "1"])) if "I_ALIVE" in df.columns else (~df["I_Alive"].astype(str).str.lower().isin(["true", "1"]))
    y = y.astype(bool).values

    # Policy laden und Feature-Selektion strikt nach Policy
    policy = load_policy()
    feature_cols = choose_policy_features(df, policy)
    if not feature_cols:
        raise RuntimeError("Keine geeigneten engineered Features in CUSTOMER_DETAILS gefunden.")
    X_df = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X_df.values

    # Modell
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Modell: wenn vorhandene RF-Probability existiert, nutze Surrogat-Probabilitätsmodell
    if "Churn_Wahrscheinlichkeit" in df.columns:
        try:
            y_prob_target = pd.to_numeric(df["Churn_Wahrscheinlichkeit"], errors="coerce").fillna(0.0).astype(float).values
            model = build_surrogate_prob_model().fit(Xs, y_prob_target)
        except Exception:
            model = build_model().fit(Xs, y)
    else:
        model = build_model().fit(Xs, y)
    tau = model.threshold()

    # p_old bestimmen (bevorzugt aus backtest_results, sonst Fallback)
    # Quelle: JSON-DB Tabelle "backtest_results" (Feld churn_probability) je Kunde und experiment_id
    try:
        db_bt = ChurnJSONDatabase()
        bt_records = (db_bt.data.get("tables", {}).get("backtest_results", {}) or {}).get("records", []) or []
        probs_by_kunde = {}
        for r in bt_records:
            try:
                if int(r.get("id_experiments") or r.get("experiment_id") or -1) != int(experiment_id):
                    continue
                k = r.get("Kunde")
                p = r.get("churn_probability") or r.get("CHURN_PROBABILITY")
                if k is not None and p is not None:
                    probs_by_kunde[int(k)] = float(p)
            except Exception:
                continue
        if "Kunde" in df.columns and probs_by_kunde:
            df["churn_probability"] = df["Kunde"].map(lambda x: probs_by_kunde.get(int(x)) if pd.notna(x) else np.nan).astype(float)
        else:
            df["churn_probability"] = np.nan
    except Exception:
        df["churn_probability"] = np.nan

    # Fallback: versuche alte Spalte "Churn_Wahrscheinlichkeit" zu verwenden, falls vorhanden
    if df["churn_probability"].isna().all() and "Churn_Wahrscheinlichkeit" in df.columns:
        try:
            df["churn_probability"] = pd.to_numeric(df["Churn_Wahrscheinlichkeit"], errors="coerce")
        except Exception:
            pass

    # Modell-basierte Wahrscheinlichkeit (surrogate), als zusätzlicher Fallback
    try:
        p_old_model = model.predict_proba(Xs)[:, 1]
    except Exception:
        p_old_model = np.zeros(Xs.shape[0])

    # Selektion: bevorzugt nach backtest churn_probability (0.4..0.9), sonst nach Modellprobabilität
    p_old_df = df["churn_probability"].astype(float).values
    sel_idx = [i for i in range(len(p_old_df)) if (not np.isnan(p_old_df[i]) and 0.4 <= float(p_old_df[i]) <= 0.9)]
    if not sel_idx:
        # Fallback-Filter mit Modellprobabilität (0.4..0.9)
        sel_idx = [i for i in range(len(p_old_model)) if 0.4 <= float(p_old_model[i]) <= 0.9]
    if not sel_idx:
        # Letzter Fallback: Top-N nach Modellprobabilität
        order = np.argsort(-p_old_model)
        n = int(limit) if (limit and int(limit) > 0) else min(200, len(order))
        sel_idx = list(order[:n])
    if limit and limit > 0:
        sel_idx = sel_idx[:int(limit)]

    # CF Suche
    individual: List[Dict[str, Any]] = []
    names = list(feature_cols)
    for idx in sel_idx:
        x0_raw = X[idx]
        # p_old aus backtest_results (bevorzugt) oder Modell
        if not np.isnan(p_old_df[idx]):
            p_old_df_i = float(p_old_df[idx])
        else:
            p_old_df_i = float(p_old_model[idx])
        # Zieldefinition: mindestens 20% Reduktion relativ zu p_old
        p_target = float(p_old_df_i * 0.8)
        z_raw, dist, top_changes = find_counterfactual(
            x0_raw, names, model, scaler, policy, max_iter=200, p_target=p_target, tau=None
        )
        p_new = float(model.predict_proba(scaler.transform(z_raw.reshape(1, -1)))[0, 1])
        no_cf = bool(p_new > p_target)
        rel_red = float((p_old_df_i - p_new) / p_old_df_i) if p_old_df_i > 0 else 0.0
        rec = {
            "Kunde": int(df.iloc[idx]["Kunde"]) if "Kunde" in df.columns else idx,
            "p_old": p_old_df_i,
            "p_new": p_new,
            "relative_reduction": rel_red,
            "no_cf_found": no_cf,
            "l2_weighted": float(dist),
            "top_changes": top_changes[:10]
        }
        individual.append(rec)

    # Aggregate
    agg_map: Dict[str, Dict[str, float]] = {}
    for rec in individual:
        if rec.get("no_cf_found"):
            continue
        for ch in rec.get("top_changes", []):
            a = agg_map.setdefault(ch["feature"], {"count": 0, "median_abs_delta": 0.0, "vals": []})
            a["count"] += 1
            a["vals"].append(float(ch["abs_delta"]))
    aggregate: List[Dict[str, Any]] = []
    for f, d in agg_map.items():
        vals = sorted(d["vals"]) if d["vals"] else [0.0]
        m = float(vals[len(vals) // 2])
        aggregate.append({"feature": f, "count": int(d["count"]), "median_abs_delta": m})
    aggregate.sort(key=lambda r: (r["count"], r["median_abs_delta"]), reverse=True)

    # Ergebnisse in JSON-Database persistieren (statt Datei-Export)
    try:
        db = ChurnJSONDatabase()
        def _upsert_table(table_name: str, records: List[Dict[str, Any]]):
            tbl = db.data["tables"].setdefault(table_name, {"description": "Counterfactuals", "source": "counterfactuals_cli", "records": []})
            existing: List[Dict[str, Any]] = tbl.get("records", []) or []
            # Entferne vorhandene Records für dieses Experiment
            remaining = [r for r in existing if int(r.get("id_experiments", -1)) != int(experiment_id)]
            # Anreichern um experiment_id
            for r in records:
                if isinstance(r, dict):
                    r = dict(r)
                    r["id_experiments"] = int(experiment_id)
                    remaining.append(r)
            tbl["records"] = remaining
            db.data["tables"][table_name] = tbl

        _upsert_table("cf_individual", individual)
        _upsert_table("cf_aggregate", aggregate)

        # Raw Reports: baue Mapping raw->engineered nur aus Policy-Auswahl
        raw_to_eng: Dict[str, List[str]] = {}
        for raw_name in policy.features.keys():
            lst = [c for c in feature_cols if c == raw_name or str(c).startswith(f"{raw_name}_") or (raw_name == "N_DIGITALIZATIONRATE" and str(c).startswith("N_DIGITALIZATIONRATE"))]
            if lst:
                raw_to_eng[raw_name] = lst
        ind_raw, agg_raw = build_raw_reports(individual, aggregate, raw_to_eng)
        _upsert_table("cf_individual_raw", ind_raw)
        _upsert_table("cf_aggregate_raw", agg_raw)

        # Business-Metriken berechnen und persistieren
        print("📊 Berechne Business-Metriken (Kosten, ROI, Feature-Empfehlungen)...")
        business_metrics = _create_business_metrics(individual, aggregate, policy, raw_to_eng, experiment_id)
        for table_name, records in business_metrics.items():
            _upsert_table(table_name, records)

        # Speichern
        db.save()
        print("💾 Counterfactuals in JSON-Database gespeichert (cf_individual, cf_aggregate, cf_individual_raw, cf_aggregate_raw)")
        print("💼 Business-Metriken gespeichert (cf_business_metrics, cf_feature_recommendations, cf_cost_analysis)")
        try:
            db.export_counterfactuals_to_outbox(int(experiment_id))
            print("📦 Outbox-Export (Counterfactuals) abgeschlossen")
        except Exception as _e:
            print(f"⚠️ Outbox-Export (Counterfactuals) fehlgeschlagen: {_e}")

    except Exception as e:
        print(f"❌ Fehler beim Persistieren der Counterfactuals in JSON-DB: {e}")

    # Beispiele (3) aus erfolgreichen Fällen als Sätze ausgeben
    try:
        inv = invert_mapping(mapping)
        examples: List[str] = []
        for rec in individual:
            if rec.get("no_cf_found"):
                continue
            changes = rec.get("top_changes", [])
            if not changes:
                continue
            ch = changes[0]
            eng = ch.get("feature")
            raw = inv.get(eng, eng)
            delta = ch.get("delta", 0.0)
            p_old_v = rec.get("p_old")
            p_new_v = rec.get("p_new")
            sgn = "+" if float(delta) >= 0 else ""
            examples.append(f"Kunde {rec.get('Kunde')}: {raw} {sgn}{round(float(delta), 3)} → Risiko {round(float(p_old_v), 3)} → {round(float(p_new_v), 3)}")
            if len(examples) >= 3:
                break
        if examples:
            print("\nBeispiele:")
            for s in examples:
                print(" - ", s)
    except Exception:
        pass

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactuals CLI (engineered features only)")
    p.add_argument("--experiment-id", type=int, required=True)
    p.add_argument("--sample", type=float, default=0.2, help="Optionaler Stichprobenanteil (0<sample<1)")
    p.add_argument("--limit", type=int, default=0, help="Maximale Anzahl Kunden (0 = alle)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = run(args.experiment_id, args.sample, args.limit)
    if ok:
        print("✅ Counterfactuals in JSON-Database gespeichert")
    else:
        print("⚠️ Counterfactuals konnten nicht gespeichert werden")


