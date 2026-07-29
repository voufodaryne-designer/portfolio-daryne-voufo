"""
Nettoyage et feature engineering — Bank Marketing Dataset (Kaggle/UCI)
Projet portfolio : Pilotage data d'une campagne de souscription bancaire

Utilisation :
    python nettoyage_bank_marketing.py --input bank-additional-full.csv --output bank_marketing_clean.csv

Le fichier source Kaggle est généralement encodé avec un point-virgule (;) comme
séparateur. Le script détecte automatiquement le séparateur (`;` ou `,`).
"""

import argparse
import pandas as pd
import numpy as np


def charger_donnees(chemin_fichier):
    """Charge le CSV en détectant automatiquement le séparateur."""
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        premiere_ligne = f.readline()
    separateur = ";" if premiere_ligne.count(";") > premiere_ligne.count(",") else ","

    df = pd.read_csv(chemin_fichier, sep=separateur)
    # Retire d'éventuelles guillemets résiduels dans les noms de colonnes
    df.columns = [c.strip().strip('"') for c in df.columns]
    print(f"Chargement : {df.shape[0]} lignes, {df.shape[1]} colonnes (séparateur détecté : '{separateur}')")
    return df


def auditer_qualite(df):
    """Affiche un rapide audit qualité avant nettoyage."""
    print("\n--- Audit qualité ---")
    print("Valeurs 'unknown' par colonne :")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        nb_unknown = (df[col].astype(str).str.lower() == "unknown").sum()
        if nb_unknown > 0:
            print(f"  {col} : {nb_unknown} ({nb_unknown / len(df):.1%})")

    doublons = df.duplicated().sum()
    print(f"Doublons : {doublons}")

    if "pdays" in df.columns:
        nb_jamais_contacte = (df["pdays"] == 999).sum()
        print(f"Clients jamais contactés auparavant (pdays=999) : {nb_jamais_contacte} ({nb_jamais_contacte / len(df):.1%})")


def nettoyer(df):
    """Nettoyage de base : cible binaire, valeurs unknown explicites, doublons."""
    df = df.copy()

    # Cible en binaire (0/1) pour faciliter les taux de conversion dans Power BI
    if "y" in df.columns:
        df["y_binaire"] = df["y"].map({"yes": 1, "no": 0})

    # On garde "unknown" comme catégorie explicite plutôt que de la supprimer :
    # supprimer les lignes ferait perdre trop de volume sur certaines colonnes
    # (ex: 'default' a souvent >20% de unknown). On la renomme pour plus de clarté.
    colonnes_categorielles = df.select_dtypes(include=["object", "string"]).columns
    for col in colonnes_categorielles:
        df[col] = df[col].replace({"unknown": "Non renseigné"})

    # Suppression des doublons stricts
    avant = len(df)
    df = df.drop_duplicates()
    print(f"\nDoublons supprimés : {avant - len(df)}")

    return df


def ajouter_features(df):
    """Feature engineering marketing : segments actionnables pour l'analyse."""
    df = df.copy()

    # Tranche d'âge
    bins_age = [0, 25, 35, 45, 55, 65, 120]
    labels_age = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    df["tranche_age"] = pd.cut(df["age"], bins=bins_age, labels=labels_age, right=True)

    # Canal digital vs traditionnel
    if "contact" in df.columns:
        df["type_canal"] = df["contact"].map({
            "cellular": "Digital (mobile)",
            "telephone": "Traditionnel (fixe)",
        }).fillna("Non renseigné")

    # Niveau d'effort commercial, basé sur le nombre de contacts pendant la campagne
    if "campaign" in df.columns:
        def niveau_effort(nb_contacts):
            if nb_contacts <= 1:
                return "1 contact"
            elif nb_contacts <= 3:
                return "2-3 contacts"
            else:
                return "4+ contacts"
        df["niveau_effort_commercial"] = df["campaign"].apply(niveau_effort)

    # Client déjà contacté lors d'une campagne précédente (pdays = 999 = jamais)
    if "pdays" in df.columns:
        df["deja_contacte_avant"] = df["pdays"].apply(lambda x: "Non" if x == 999 else "Oui")

    # Segment basé sur l'historique de la campagne précédente
    if "poutcome" in df.columns:
        df["segment_historique"] = df["poutcome"].map({
            "success": "Ancien converti",
            "failure": "Ancien refus",
            "nonexistent": "Nouveau client",
        }).fillna("Non renseigné")

    return df


def resume_final(df):
    """Affiche un résumé des taux de conversion clés pour vérification rapide."""
    if "y_binaire" not in df.columns:
        return
    print("\n--- Résumé taux de conversion ---")
    print(f"Taux de conversion global : {df['y_binaire'].mean():.1%}")

    if "type_canal" in df.columns:
        print("\nPar canal :")
        print(df.groupby("type_canal")["y_binaire"].mean().apply(lambda x: f"{x:.1%}"))

    if "niveau_effort_commercial" in df.columns:
        print("\nPar niveau d'effort commercial :")
        print(df.groupby("niveau_effort_commercial")["y_binaire"].mean().apply(lambda x: f"{x:.1%}"))

    if "segment_historique" in df.columns:
        print("\nPar segment historique :")
        print(df.groupby("segment_historique")["y_binaire"].mean().apply(lambda x: f"{x:.1%}"))


def main():
    parser = argparse.ArgumentParser(description="Nettoyage du dataset Bank Marketing")
    parser.add_argument("--input", required=True, help="Chemin du fichier CSV source (ex: bank-additional-full.csv)")
    parser.add_argument("--output", default="bank_marketing_clean.csv", help="Chemin du fichier CSV nettoyé en sortie")
    args = parser.parse_args()

    df = charger_donnees(args.input)
    auditer_qualite(df)
    df = nettoyer(df)
    df = ajouter_features(df)
    resume_final(df)

    df.to_csv(args.output, index=False)
    print(f"\nFichier nettoyé exporté : {args.output} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    print("Prêt à importer dans Power BI.")


if __name__ == "__main__":
    main()
