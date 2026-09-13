from langgraph.graph import StateGraph, END
from src.orchestration.state import PipelineState


def construire_graph(generateur, validateur, curateur):
    def noeud_generer(state: PipelineState) -> PipelineState:
        tentative = state["tentative"] + 1
        print(f"\n[Graph] --- Génération (tentative {tentative}/{state['max_tentatives']}) ---")

        df_batch = generateur.generer_batch(
            state["df_source"],
            nb_lignes=state["nb_lignes"],
            nb_seeds=state["nb_seeds"],
            persona=state["persona_actuel"],           # none au 1er passage -> choisi aleatoirement
            erreur_precedente=state["erreur_validation"],  # None si 1ere tentative
        )

        return {
            **state,
            "df_batch": df_batch,
            "tentative": tentative,
            "persona_actuel": generateur.dernier_persona_utilise,
            "erreur_validation": None,
        }

    def noeud_valider_format(state: PipelineState) -> PipelineState:
        if state["df_batch"] is None or state["df_batch"].empty:
            return {
                **state,
                "erreur_validation": "Le générateur n'a produit aucune ligne exploitable (échec API ou JSON invalide).",
            }

        df_valide, ok = validateur.valider_format(state["df_batch"], state["df_source"])
        if ok:
            return {**state, "df_batch": df_valide, "erreur_validation": None}

        return {
            **state,
            "erreur_validation": validateur.dernier_erreur_format or "Erreur de format inconnue.",
        }

    def noeud_valider_semantique(state: PipelineState) -> PipelineState:
        df_avant = state["df_batch"]
        df_nettoye = validateur.valider_semantique(df_avant, seuil_similarite=state["seuil_similarite"])

        if df_nettoye.empty and not df_avant.empty:
            return {
                **state,
                "df_batch": df_nettoye,
                "erreur_validation": (
                    "Toutes les lignes générées étaient des quasi-doublons de données déjà "
                    "existantes (dataset source ou batches précédents). Varie beaucoup plus "
                    "les valeurs, le style et les combinaisons."
                ),
                "statut": "invalide_semantique",
            }

        return {**state, "df_batch": df_nettoye, "statut": "valide"}

    def noeud_curer(state: PipelineState) -> PipelineState:
        nb = curateur.recevoir_batch_valide(state["df_batch"])
        print(f"[Graph] --- Curateur : {nb} lignes acceptées dans le dataset final ---")
        return {**state, "statut": "valide"}

    def noeud_echec(state: PipelineState) -> PipelineState:
        print(f"[Graph] --- Abandon du batch après {state['tentative']} tentatives : {state['erreur_validation']} ---")
        return {**state, "statut": "echec"}

    def router_apres_format(state: PipelineState) -> str:
        if state["erreur_validation"] is not None:
            if state["tentative"] >= state["max_tentatives"]:
                return "echec"
            return "regenerer"
        return "continuer"

    def router_apres_semantique(state: PipelineState) -> str:
        if state["erreur_validation"] is not None:
            if state["tentative"] >= state["max_tentatives"]:
                return "echec"
            return "regenerer"
        return "termine"

    graphe = StateGraph(PipelineState)

    graphe.add_node("generer", noeud_generer)
    graphe.add_node("valider_format", noeud_valider_format)
    graphe.add_node("valider_semantique", noeud_valider_semantique)
    graphe.add_node("curer", noeud_curer)
    graphe.add_node("echec", noeud_echec)

    graphe.set_entry_point("generer")
    graphe.add_edge("generer", "valider_format")

    graphe.add_conditional_edges(
        "valider_format",
        router_apres_format,
        {"continuer": "valider_semantique", "regenerer": "generer", "echec": "echec"},
    )

    graphe.add_conditional_edges(
        "valider_semantique",
        router_apres_semantique,
        {"termine": "curer", "regenerer": "generer", "echec": "echec"},
    )

    graphe.add_edge("curer", END)
    graphe.add_edge("echec", END)

    return graphe.compile()
