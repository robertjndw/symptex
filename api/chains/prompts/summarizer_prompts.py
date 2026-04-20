from langchain_core.prompts import SystemMessagePromptTemplate, MessagesPlaceholder, ChatPromptTemplate

def get_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            Du erhältst als letzte Nachricht den vollständigen Rohtext aus medizinischen Dokumenten.
            Deine Aufgabe: Erstelle eine kurze, sachliche, streng extraktive Zusammenfassung.
            
            Regeln:
        
            Nur Informationen verwenden, die wörtlich im Eingangstext stehen.
            Nichts erfinden. Keine Interpretation. Kein Fachwissen ergänzen.
            Keine eigenen Diagnosen ableiten. Diagnosen nur nennen, wenn sie explizit im Text stehen.
            Steht etwas nicht im Text, gilt es als unbekannt.
            
            Stil:
            
            Kurz, klar, strukturiert.
            Nur relevante Inhalte zusammenfassen, wie sie genau im Text beschrieben sind.
            Falls keine medizinischen Informationen enthalten sind: „Keine auswertbaren medizinischen Informationen.“
            """),
        MessagesPlaceholder(variable_name="messages"), ])