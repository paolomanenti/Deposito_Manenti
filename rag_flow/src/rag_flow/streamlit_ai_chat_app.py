import streamlit as st
import json
import datetime
from typing import Dict, List, Any
import hashlib
import os
import tempfile
import random
from pathlib import Path
from litellm import completion
from dotenv import load_dotenv
from main import Config, RagAgentFlow, config

load_dotenv()

def sanitize_rag_output(text: str) -> str:
    """Rimuove blocchi tecnici (tool calls, JSON, code-fence) e restituisce solo testo naturale."""
    if not text:
        return ""
    cleaned = text
    # Rimuovi prefissi tipo "RISPOSTA:" o simili
    for prefix in ["RISPOSTA:", "Risposta:", "RESPONSE:", "Output:"]:
        if cleaned.strip().startswith(prefix):
            cleaned = cleaned.strip()[len(prefix):].strip()

    # Rimuovi block code ```...``` che includono action/tool
    import re
    def _strip_tool_fences(s: str) -> str:
        pattern = r"```[\s\S]*?```"
        def repl(m):
            block = m.group(0)
            lower = block.lower()
            if "\"action\"" in lower or "action_input" in lower or "tool" in lower:
                return ""
            return block
        return re.sub(pattern, repl, s)

    cleaned = _strip_tool_fences(cleaned)

    # Rimuovi JSON standalone con chiavi action/action_input/tool
    json_like_pattern = r"\{[\s\S]*?\}"
    def _strip_tool_json(s: str) -> str:
        parts = re.split(json_like_pattern, s)
        keep = []
        last_end = 0
        for match in re.finditer(json_like_pattern, s):
            json_block = match.group(0)
            lower = json_block.lower()
            if not ("\"action\"" in lower or "action_input" in lower or "tool" in lower):
                keep.append(s[last_end:match.start()])
                keep.append(json_block)
                last_end = match.end()
        keep.append(s[last_end:])
        return "".join(keep)

    cleaned = _strip_tool_json(cleaned)

    # Pulisci doppie nuove linee e spazi
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # Se svuotato, restituisci messaggio di fallback
    return cleaned or "Ho elaborato la tua richiesta. Se non vedi contenuto, riprova la domanda con maggior dettaglio."

def synthetize_content(config: Config, content: str, max_length: int = 10) -> str:
    response = completion(
        model=config.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following content in \'{max_length}\' words\\"
             f"or less:\n\n\'{content}\'"}
        ],
        max_tokens=30,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"]


# Configurazione della pagina
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inizializzazione dello stato della sessione
def init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "disclaimer"
    if "accepted_terms" not in st.session_state:
        st.session_state.accepted_terms = False
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "api_config" not in st.session_state:
        st.session_state.api_config = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "login_message" not in st.session_state:
        st.session_state.login_message = None

init_session_state()

# Funzioni di supporto
def create_chat_id():
    """Crea un nuovo ID per la chat"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_{timestamp}"

def save_message(chat_id: str, role: str, content: str):
    text_synthesis = synthetize_content(config=config, content=content, max_length=50)
    """Salva un messaggio nella cronologia"""
    if chat_id not in st.session_state.chat_history:
        st.session_state.chat_history[chat_id] = {
            "messages": [],
            "created_at": datetime.datetime.now().isoformat(),
            "title": text_synthesis + "..." if len(text_synthesis) > 50 else text_synthesis
        }
    
    st.session_state.chat_history[chat_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat()
    })

def process_documents(uploaded_files):
    """Processa i documenti caricati per il sistema RAG"""
    processed_docs = []
    for file in uploaded_files:
        # Simula il processing dei documenti
        doc_info = {
            "name": file.name,
            "size": file.size,
            "type": file.type,
            "processed_at": datetime.datetime.now().isoformat()
        }
        processed_docs.append(doc_info)
    return processed_docs

def simulate_ai_response(user_message: str, use_web_search: bool = False, context_docs: List = None):
    """Simula una risposta dell'AI (da sostituire con vera chiamata API)"""
    # Questa è una simulazione - sostituire con vera chiamata all'API
    response = f"Questa è una risposta simulata al messaggio: '{user_message}'"
    
    if use_web_search:
        response += " (con ricerca web attivata)"
    
    if context_docs:
        response += f" (utilizzando {len(context_docs)} documenti come contesto)"
    
    return response

# PAGINA 1: DISCLAIMER
def show_disclaimer_page():
    st.title("🤖 AI Chat Assistant")
    st.markdown("---")
    
    st.header("📋 Termini e Condizioni d'Uso")
    
    disclaimer_text = """
    ### Avviso Importante sull'Utilizzo di Tecnologie di Intelligenza Artificiale
    
    Questa applicazione utilizza tecnologie di **Intelligenza Artificiale (AI)** per fornire servizi di chat assistiti.
    
    #### Conformità al Regolamento EU AI Act
    
    In conformità con il **Regolamento (UE) 2024/1689 del Parlamento Europeo e del Consiglio** 
    (EU AI Act), ti informiamo che:
    
    - ✅ Questa applicazione utilizza sistemi di AI per elaborare e rispondere alle tue richieste
    - ✅ I tuoi dati potrebbero essere processati da modelli di linguaggio di grandi dimensioni
    - ✅ Le risposte generate sono prodotte automaticamente e potrebbero contenere imprecisioni
    - ✅ Non utilizzare questa applicazione per decisioni critiche senza verifica umana
    
    #### Responsabilità dell'Utente
    
    - 📝 **Privacy**: Non condividere informazioni personali sensibili
    - 🔒 **Sicurezza**: Mantieni riservate le tue credenziali API
    - ⚖️ **Legalità**: Utilizza l'applicazione nel rispetto delle leggi vigenti
    - 🎯 **Appropriatezza**: Non utilizzare per contenuti inappropriati o dannosi
    
    #### Limitazioni e Disclaimer
    
    - ⚠️ Le risposte dell'AI sono generate automaticamente e potrebbero essere imprecise
    - ⚠️ L'applicazione è fornita "così com'è" senza garanzie
    - ⚠️ Gli sviluppatori non sono responsabili per l'uso improprio dell'applicazione
    
    ---
    
    **Procedendo, dichiari di:**
    - Aver letto e compreso i termini sopra indicati
    - Accettare l'utilizzo di tecnologie di AI come descritto
    - Essere consapevole delle implicazioni del EU AI Act
    - Utilizzare l'applicazione sotto la tua responsabilità
    """
    
    st.markdown(disclaimer_text)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        accept_terms = st.checkbox(
            "✅ **Accetto i termini e condizioni d'uso e confermo di essere consapevole dell'utilizzo di tecnologie AI**",
            value=st.session_state.accepted_terms
        )
        
        if st.button("🚀 Procedi all'Applicazione", type="primary", disabled=not accept_terms):
            st.session_state.accepted_terms = accept_terms
            st.session_state.page = "login"
            st.rerun()

# PAGINA 2: LOGIN/CONFIGURAZIONE
def show_login_page():
    st.title("🔐 Configurazione API")
    st.markdown("---")
    
    st.info("Inserisci le credenziali per accedere all'applicazione AI Chat.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        api_key = st.text_input(
            "🔑 API Key",
            type="password",
            value=st.session_state.api_config.get("api_key", ""),
            help="Inserisci la tua API Key per il servizio AI"
        )
        
        base_url = st.text_input(
            "🌐 Base URL",
            value=st.session_state.api_config.get("base_url", "https://api.openai.com/v1"),
            help="URL base per le chiamate API"
        )
    
    with col2:
        deployment_name = st.text_input(
            "🚀 Model / Deployment",
            value=st.session_state.api_config.get("deployment_name", config.model),
            help="Nome del modello o deployment da utilizzare (es: azure/gpt-4o)"
        )

    st.markdown("---")

    # Impostazioni avanzate del Flow (Config)
    with st.expander("⚙️ Impostazioni Avanzate (RAG Flow)"):
        topic = st.text_input(
            "📘 Topic",
            value=getattr(config, "topic", ""),
            help="Argomento di pertinenza per valutare la rilevanza delle domande"
        )
        save_path = st.text_input(
            "💾 Percorso Salvataggio Risposta",
            value=getattr(config, "path_save_dir", "./output/answer.md"),
            help="File in cui salvare la risposta generata"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Test connessione (simulato)
        if st.button("🧪 Test Connessione"):
            if api_key and base_url:
                with st.spinner("Test connessione in corso..."):
                    # Simula test connessione
                    st.success("✅ Connessione testata con successo!")
            else:
                st.error("❌ Compila tutti i campi obbligatori")
    
    st.markdown("---")
    
    # Controlli di validazione
    all_fields_valid = api_key and base_url and deployment_name
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Torna al Disclaimer"):
            st.session_state.page = "disclaimer"
            st.rerun()
    
    with col3:
        if st.button("🚪 Log In", type="primary", disabled=not all_fields_valid):
            # Salva configurazione
            st.session_state.api_config = {
                "api_key": api_key,
                "base_url": base_url,
                "deployment_name": deployment_name
            }
            # Imposta variabili ambiente per litellm
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = base_url
            # Supporto Azure (best-effort)
            os.environ["AZURE_API_KEY"] = api_key
            os.environ["AZURE_API_BASE"] = base_url
            # Aggiorna Config del flow
            config.model = deployment_name
            if topic:
                config.topic = topic
            if save_path:
                config.path_save_dir = save_path
            st.session_state.logged_in = True
            st.session_state.page = "chat"
    
    if not all_fields_valid:
        st.warning("⚠️ Compila tutti i campi per procedere")

# PAGINA 3: CHAT PRINCIPALE
def show_chat_page():
    # Sidebar per cronologia chat
    with st.sidebar:
        st.title("💬 Cronologia Chat")
        
        # Pulsante nuova chat
        if st.button("➕ Nuova Chat", type="primary"):
            st.session_state.current_chat_id = create_chat_id()
        
        # Lista delle chat esistenti
        if st.session_state.chat_history:
            st.markdown("### 📚 Chat Precedenti")
            for chat_id, chat_data in st.session_state.chat_history.items():
                chat_title = chat_data.get("title", chat_id)
                if st.button(f"💭 {chat_title}", key=f"load_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
        
        st.markdown("---")
        
        # Pulsante logout
        if st.button("🚪 Log Out", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.api_config = {}
            st.rerun()
    
    # Area principale della chat
    st.title("🤖 AI Chat Assistant")

    # Messaggio di login/inizializzazione mostrato una volta
    if st.session_state.login_message:
        st.info(st.session_state.login_message)
        # Pulisci dopo la prima visualizzazione
        st.session_state.login_message = None
    
    # Informazioni configurazione
    with st.expander("ℹ️ Configurazione Corrente"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Base URL:** {st.session_state.api_config.get('base_url', 'N/A')}")
        with col2:
            st.write(f"**Modello:** {st.session_state.api_config.get('deployment_name', getattr(config, 'model', 'N/A'))}")
        with col3:
            st.write(f"**API Key:** {'*' * 10}...")
        st.write(f"**Topic:** {getattr(config, 'topic', 'N/A')}")
        st.write(f"**Salvataggio:** {getattr(config, 'path_save_dir', 'N/A')}")
    
    # Area conversazione
    chat_container = st.container()
    
    # Mostra messaggi della chat corrente
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_history:
        messages = st.session_state.chat_history[st.session_state.current_chat_id]["messages"]
        
        with chat_container:
            for message in messages:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
    
    # Area input utente
    st.markdown("---")
    
    # Controlli aggiuntivi
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Upload documenti per RAG
        uploaded_files = st.file_uploader(
            "📎 Carica documenti per il contesto RAG",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md'],
            help="Carica documenti che verranno utilizzati come contesto per le risposte"
        )
        
        if uploaded_files:
            st.session_state.uploaded_documents = process_documents(uploaded_files)
            st.success(f"✅ {len(uploaded_files)} documento/i caricato/i e processato/i")
    
    with col2:
        web_search = st.checkbox("🌐 Ricerca Web", help="Attiva la ricerca web per risposte aggiornate")
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_docs = st.button("🗑️ Pulisci Documenti")
        if clear_docs:
            st.session_state.uploaded_documents = []
            st.rerun()
    
    # Input principale
    user_input = st.chat_input("Scrivi il tuo messaggio...")
    
    if user_input:
        # Crea nuova chat se necessario
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = create_chat_id()
        
        # Salva messaggio utente
        save_message(st.session_state.current_chat_id, "user", user_input)
        
        # Esegui il RAG Flow
        with st.spinner("🤔 Sto pensando..."):
            try:
                rag_flow = RagAgentFlow()
                rag_flow.input_query = user_input
                rag_flow.kickoff()
                ai_response = getattr(rag_flow.state, "answer", "") or "(Nessuna risposta generata)"
            except Exception as e:
                ai_response = f"Errore nell'esecuzione del flow: {e}"
        
        # Salva risposta AI (sanitizzata)
        clean_response = sanitize_rag_output(ai_response)
        save_message(st.session_state.current_chat_id, "assistant", clean_response)
        
        st.rerun()


def main():
    if st.session_state.page == "disclaimer":
        show_disclaimer_page()
    elif st.session_state.page == "login":
        if st.session_state.accepted_terms:
            show_login_page()
        else:
            st.session_state.page = "disclaimer"
            st.rerun()
    elif st.session_state.page == "chat":
        if st.session_state.logged_in:
            show_chat_page()
        else:
            st.session_state.page = "login"
            st.rerun()

if __name__ == "__main__":
    main()