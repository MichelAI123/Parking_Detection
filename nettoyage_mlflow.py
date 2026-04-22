import mlflow

def forcer_fermeture_sessions():
    """
    Script utilitaire de secours : Ferme toute session MLflow restée ouverte par erreur.
    À n'utiliser qu'en phase de développement, jamais en production.
    """
    if mlflow.active_run():
        print(f"Fermeture forcée de la session active : {mlflow.active_run().info.run_id}")
        mlflow.end_run()
    else:
        print("Aucune session MLflow fantôme n'est actuellement ouverte.")

if __name__ == "__main__":
    forcer_fermeture_sessions()