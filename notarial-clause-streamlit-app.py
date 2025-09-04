# Replace the load_dotenv() section with:
import streamlit as st
import os

# For Streamlit Cloud deployment
if 'GEMINI_API_KEY' in st.secrets:
    os.environ['GEMINI_API_KEY'] = st.secrets['GEMINI_API_KEY']
    
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import PyPDF2
from pathlib import Path
import json
from datetime import datetime
import re
import tempfile

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Notariële Clausule Processor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'notarial_info' not in st.session_state:
    st.session_state.notarial_info = {}
if 'processed_clauses' not in st.session_state:
    st.session_state.processed_clauses = {}
if 'source_content' not in st.session_state:
    st.session_state.source_content = ""
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'intake'
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = {}
if 'current_questions' not in st.session_state:
    st.session_state.current_questions = []
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0

# Configure Gemini API
if os.getenv('GEMINI_API_KEY'):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
else:
    st.error("⚠️ GEMINI_API_KEY not found in environment variables!")
    st.stop()

# Essential clauses that should always be kept
ESSENTIAL_CLAUSES = [
    1, 2, 3, 4, 5, 6, 7,  # 1-7
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,  # 15-27
    33, 34, 35, 36,  # 33-36
    38, 40, 41, 44,  # 38, 40, 41, 44
    47, 48, 49, 50,  # 47-50
    52,  # 52
    54, 55, 56, 57, 58, 59, 60  # 54-60
]

# ============= HELPER FUNCTIONS FROM ORIGINAL SCRIPT =============

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        
    return text

def load_source_documents(uploaded_files):
    """Load content from uploaded files"""
    combined_content = ""
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                content = extract_text_from_pdf(tmp_path)
                if content:
                    combined_content += f"\n\n--- Content from {uploaded_file.name} ---\n\n"
                    combined_content += content
            elif uploaded_file.name.lower().endswith(('.txt', '.text')):
                content = uploaded_file.getvalue().decode('utf-8')
                combined_content += f"\n\n--- Content from {uploaded_file.name} ---\n\n"
                combined_content += content
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    return combined_content

def get_dutch_month(month_num):
    """Convert month number to Dutch month name"""
    months = {
        1: "januari", 2: "februari", 3: "maart", 4: "april",
        5: "mei", 6: "juni", 7: "juli", 8: "augustus",
        9: "september", 10: "oktober", 11: "november", 12: "december"
    }
    return months.get(month_num, "")

def extract_info_from_documents(source_content):
    """Use Gemini to extract notarial information from source documents"""
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    extraction_prompt = """
    Analyseer de volgende documenten en extraheer alle relevante notariële informatie.
    LET OP: Er kunnen MEERDERE verkopers en/of kopers zijn. Detecteer het exacte aantal.
    
    Analyseer ook:
    - Type verkoper (alleenstaand/gehuwd koppel/samenwonend/vennootschap)
    - Type koper (alleenstaand/gehuwd koppel/samenwonend/vennootschap)
    - Wijze van aankoop (volle eigendom/vruchtgebruik/met aanwas/zonder aanwas)
    - Wat wordt verkocht (alleen onroerend/met roerend/met of zonder meetplan)
    - Historiek (zelf gekocht/via schenking/ouders aan kind)
    
    Geef het resultaat in JSON formaat met de volgende structuur.
    Voor elk veld, geef ook een confidence score (0-100) die aangeeft hoe zeker je bent.
    Als informatie niet gevonden wordt, gebruik "NOT_FOUND" als waarde en 0 als confidence.

    {
        "algemene_info": {
            "ondertekening_datum": {"value": "DD-MM-JJJJ of NOT_FOUND", "confidence": 0-100},
            "videoconferentie": {"value": true/false/null, "confidence": 0-100}
        },
        "transactie_info": {
            "verkoper_type": {"value": "alleenstaande/gehuwd_koppel/wettelijk_samenwonend/feitelijk_samenwonend/vennootschap/NOT_FOUND", "confidence": 0-100},
            "koper_type": {"value": "alleenstaande/gehuwd_koppel/wettelijk_samenwonend/feitelijk_samenwonend/vennootschap/NOT_FOUND", "confidence": 0-100},
            "aankoop_wijze": {"value": ["volle_eigendom", "gesplitste_aankoop", "met_aanwas", "zonder_aanwas"], "confidence": 0-100},
            "verkoop_object": {"value": ["enkel_onroerend", "met_roerend", "zonder_meetplan", "met_meetplan"], "confidence": 0-100},
            "historiek": {"value": "zelf_gekocht/via_schenking/ouders_aan_kind/NOT_FOUND", "confidence": 0-100}
        },
        "aantal_partijen": {
            "aantal_verkopers": {"value": number, "confidence": 0-100},
            "aantal_kopers": {"value": number, "confidence": 0-100}
        },
        "verkopers": [
            {
                "volgnummer": 1,
                "voornaam": {"value": "naam of NOT_FOUND", "confidence": 0-100},
                "achternaam": {"value": "naam of NOT_FOUND", "confidence": 0-100},
                "rijksregisternummer": {"value": "nummer of NOT_FOUND", "confidence": 0-100},
                "geboorteplaats": {"value": "plaats of NOT_FOUND", "confidence": 0-100},
                "geboortedatum": {"value": "DD-MM-JJJJ of NOT_FOUND", "confidence": 0-100},
                "adres": {"value": "volledig adres of NOT_FOUND", "confidence": 0-100},
                "burgerlijke_staat": {"value": "staat of NOT_FOUND", "confidence": 0-100},
                "partner_naam": {"value": "naam of NOT_FOUND", "confidence": 0-100}
            }
        ],
        "kopers": [
            // Zelfde structuur als verkopers
        ],
        "onroerend_goed_info": {
            "kadastrale_gegevens": {"value": "gegevens of NOT_FOUND", "confidence": 0-100},
            "koopsom": {"value": "bedrag of NOT_FOUND", "confidence": 0-100},
            "leeg_bij_overdracht": {"value": true/false/null, "confidence": 0-100}
        }
    }

    Documenten:
    """ + source_content[:10000]
    
    try:
        response = model.generate_content(extraction_prompt)
        
        # Parse JSON response
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            extracted_data = json.loads(json_match.group())
            return extracted_data
        else:
            return {}
            
    except Exception as e:
        st.warning(f"Automatische extractie gefaald: {str(e)}")
        return {}

def format_notarial_info_as_text(info):
    """Format notarial information as text to append to source documents"""
    text = "\n\n--- NOTARIËLE INFORMATIE ---\n\n"
    
    # Algemene informatie
    text += "ALGEMENE AKTE INFORMATIE:\n"
    if 'ondertekening_datum' in info:
        text += f"- Datum ondertekening: {info['ondertekening_datum']}\n"
        text += f"- Dag: {info.get('ondertekening_dag', '')}\n"
        text += f"- Maand: {info.get('ondertekening_maand_nl', '')}\n"
        text += f"- Jaar: {info.get('ondertekening_jaar', '')}\n"
    
    if 'repertorium_nummer' in info:
        text += f"- Repertorium nummer: {info['repertorium_nummer']}\n"
    
    if 'videoconferentie' in info:
        text += f"- Videoconferentie: {'Ja' if info['videoconferentie'] else 'Nee'}\n"
    
    # Add new intake questions to notarial info text
    if 'verkoper_type' in info:
        text += f"\n- Verkoper type: {info['verkoper_type'].replace('_', ' ')}\n"
    
    if 'koper_type' in info:
        text += f"- Koper type: {info['koper_type'].replace('_', ' ')}\n"
    
    if 'aankoop_wijze' in info:
        text += f"- Wijze van aankoop: {', '.join(info['aankoop_wijze']).replace('_', ' ')}\n"
    
    if 'verkoop_object' in info:
        text += f"- Verkoop object: {', '.join(info['verkoop_object']).replace('_', ' ')}\n"
    
    if 'historiek' in info:
        text += f"- Historiek: {info['historiek'].replace('_', ' ')}\n"
    
    # Verkopers
    if 'verkopers' in info:
        text += f"\nVERKOPERS (aantal: {len(info['verkopers'])}):\n"
        for verkoper in info['verkopers']:
            text += f"\nVerkoper {verkoper['volgnummer']}:\n"
            if 'voornaam' in verkoper and 'achternaam' in verkoper:
                text += f"- Naam: {verkoper['voornaam']} {verkoper['achternaam']}\n"
            if 'rijksregisternummer' in verkoper:
                text += f"- Rijksregisternummer: {verkoper['rijksregisternummer']}\n"
            if 'adres' in verkoper:
                text += f"- Adres: {verkoper['adres']}\n"
            if 'burgerlijke_staat' in verkoper:
                text += f"- Burgerlijke staat: {verkoper['burgerlijke_staat']}\n"
            if verkoper.get('partner_naam'):
                text += f"- Partner: {verkoper['partner_naam']}\n"
            
            # Aanwezigheid
            aanwezig_info = next((v for v in info.get('verkopers_aanwezig', []) 
                                if v['volgnummer'] == verkoper['volgnummer']), None)
            if aanwezig_info:
                text += f"- Persoonlijk aanwezig: {'Ja' if aanwezig_info['aanwezig'] else 'Nee'}\n"
    
    # Kopers
    if 'kopers' in info:
        text += f"\nKOPERS (aantal: {len(info['kopers'])}):\n"
        for koper in info['kopers']:
            text += f"\nKoper {koper['volgnummer']}:\n"
            if 'voornaam' in koper and 'achternaam' in koper:
                text += f"- Naam: {koper['voornaam']} {koper['achternaam']}\n"
            if 'rijksregisternummer' in koper:
                text += f"- Rijksregisternummer: {koper['rijksregisternummer']}\n"
            if 'adres' in koper:
                text += f"- Adres: {koper['adres']}\n"
            if 'burgerlijke_staat' in koper:
                text += f"- Burgerlijke staat: {koper['burgerlijke_staat']}\n"
            if koper.get('partner_naam'):
                text += f"- Partner: {koper['partner_naam']}\n"
            
            # Aanwezigheid
            aanwezig_info = next((k for k in info.get('kopers_aanwezig', []) 
                                if k['volgnummer'] == koper['volgnummer']), None)
            if aanwezig_info:
                text += f"- Persoonlijk aanwezig: {'Ja' if aanwezig_info['aanwezig'] else 'Nee'}\n"
    
    return text

# ============= AGENT FUNCTIONS =============

def research_agent_determine_needs(prompt, clause_type, source_content, model):
    """Research agent that determines what information is needed"""
    escaped_prompt = prompt.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    escaped_source_content = source_content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    research_prompt = f"""You are a legal research agent. Your task is to:
1. Analyze what information is needed to properly answer the given prompt
2. Search for this information in the provided documents
3. Extract all relevant information found
4. DETERMINE WHICH SCENARIO APPLIES based on the found information

CLAUSE TYPE: {clause_type}

PROMPT TO ANSWER:
{escaped_prompt}

SOURCE DOCUMENTS:
{escaped_source_content[:10000]}

IMPORTANT: If the prompt contains conditional blocks (like [BLOCK ALLEN_AANWEZIG] vs [BLOCK MET_VERTEGENWOORDIGING]), 
determine which scenario applies based on the actual situation in the documents.

Please:
1. First determine ALL information needed to create a complete legal clause based on the prompt
2. Search thoroughly through the documents for each piece of information
3. Consider variations in terminology and indirect references
4. Extract exact quotes or specific information when found
5. DETERMINE THE APPLICABLE SCENARIO based on your findings

Respond in JSON format (IN DUTCH/NEDERLANDS):
{{
    "applicable_scenario": "welk scenario van toepassing is (bijv. 'allen aanwezig' of 'met vertegenwoordiging')",
    "required_information": [
        {{
            "item": "beschrijving van benodigde informatie",
            "importance": "CRITICAL/HIGH/MEDIUM",
            "typical_location": "waar dit normaal te vinden is"
        }}
    ],
    "found_information": {{
        "item_key": {{
            "value": "gevonden waarde",
            "source_quote": "exacte quote uit document",
            "confidence": "HIGH/MEDIUM/LOW"
        }}
    }},
    "missing_information": [
        {{
            "item": "ontbrekende informatie",
            "searched_terms": ["zoektermen gebruikt"],
            "required_for": "waarom dit nodig is"
        }}
    ],
    "research_summary": "samenvatting van het onderzoek"
}}"""

    try:
        response = model.generate_content(research_prompt)
        result_text = response.text
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                # Try to clean common JSON issues
                cleaned_json = json_match.group()
                cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
                cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
                try:
                    return json.loads(cleaned_json)
                except:
                    return {
                        "applicable_scenario": "Unknown",
                        "required_information": [],
                        "found_information": {},
                        "missing_information": [],
                        "research_summary": "JSON parsing failed"
                    }
        else:
            return {
                "applicable_scenario": "Unknown",
                "required_information": [],
                "found_information": {},
                "missing_information": [],
                "research_summary": "Research parsing failed - no JSON found"
            }
    except Exception as e:
        return {
            "applicable_scenario": "Unknown",
            "required_information": [],
            "found_information": {},
            "missing_information": [],
            "research_summary": f"Research error: {str(e)}"
        }

def check_clause_applicability(prompt, clause_type, skip_conditions, source_content, notarial_info, model):
    """Applicability Agent that checks if a clause should be skipped"""
    clause_text = prompt
    
    # Format notarial info as "Klantinformatie"
    klantinfo_text = "\n[Klantinformatie]\n"
    
    # Basic info
    if 'verkopers' in notarial_info:
        klantinfo_text += f"Aantal verkopers: {len(notarial_info['verkopers'])}\n"
    if 'kopers' in notarial_info:
        klantinfo_text += f"Aantal kopers: {len(notarial_info['kopers'])}\n"
    
    # Detailed seller info
    for verkoper in notarial_info.get('verkopers', []):
        num = verkoper['volgnummer']
        klantinfo_text += f"Voornaam verkoper {num}: {verkoper.get('voornaam', 'Onbekend')}\n"
        klantinfo_text += f"Achternaam verkoper {num}: {verkoper.get('achternaam', 'Onbekend')}\n"
        klantinfo_text += f"Burgerlijke staat verkoper {num}: {verkoper.get('burgerlijke_staat', 'Onbekend')}\n"
        
        # Check aanwezigheid
        aanwezig_info = next((v for v in notarial_info.get('verkopers_aanwezig', []) 
                            if v['volgnummer'] == num), None)
        if aanwezig_info:
            klantinfo_text += f"Is verkoper {num} persoonlijk aanwezig? (ja/nee): {'ja' if aanwezig_info['aanwezig'] else 'nee'}\n"
    
    # Detailed buyer info
    for koper in notarial_info.get('kopers', []):
        num = koper['volgnummer']
        klantinfo_text += f"Voornaam koper {num}: {koper.get('voornaam', 'Onbekend')}\n"
        klantinfo_text += f"Achternaam koper {num}: {koper.get('achternaam', 'Onbekend')}\n"
        klantinfo_text += f"Burgerlijke staat koper {num}: {koper.get('burgerlijke_staat', 'Onbekend')}\n"
        
        # Check aanwezigheid
        aanwezig_info = next((k for k in notarial_info.get('kopers_aanwezig', []) 
                            if k['volgnummer'] == num), None)
        if aanwezig_info:
            klantinfo_text += f"Is koper {num} persoonlijk aanwezig? (ja/nee): {'ja' if aanwezig_info['aanwezig'] else 'nee'}\n"
    
    # Other relevant info
    if 'videoconferentie' in notarial_info:
        klantinfo_text += f"Videoconferentie: {'ja' if notarial_info['videoconferentie'] else 'nee'}\n"
    
    check_prompt = f"""Je bent een gespecialiseerde AI-assistent voor notarieel werk in België. Jouw taak is om een voorgelegde clausule te analyseren en te bepalen of deze volledig verwijderd moet worden. Je redeneert als een ervaren medewerker: feitelijk onjuiste clausules worden verwijderd, maar relevante juridische opties voor de cliënten worden behouden in de ontwerpakte.

GOUDEN REGEL: HET DOSSIER IS DE VOLLEDIGE EN ENIGE WAARHEID
Je analyseert enkel en alleen de informatie in [Klantinformatie] en [Documenten]. Je hanteert het onwrikbare principe dat dit dossier 100% compleet is. De afwezigheid van informatie over een feit (bv. Bosdecreet, stookolietank) is voor jou het definitieve bewijs dat dit feit niet van toepassing is.

CLAUSULE KENNISBANK
Je categoriseert elke clausule en past de bijbehorende logica strikt toe.

Categorie 1a: Feitelijk Bepaalde Clausules
Logica: SCHRAPPEN, tenzij er positief bewijs in het dossier is.
- Clausule 26 "BOSDECREET": Schrappen, tenzij een document vermeldt dat het goed onder het Bosdecreet valt.
- Clausules 28-32 "Specifieke Voorkooprechten": Schrappen, tenzij een document de toepassing bevestigt.
- Clausule 37 "RISICOGROND": Schrappen, tenzij een document de grond als 'risicogrond' classificeert.
- Clausule 39 "STOOKOLIETANKS": Schrappen, tenzij het dossier de aanwezigheid vermeldt.
- Clausule 45 "ALARMINSTALLATIE": Schrappen, tenzij het dossier de aanwezigheid vermeldt.
- Clausule 46 "ZONNEPANELEN": Schrappen, tenzij het dossier de aanwezigheid vermeldt.
- Clausule 53 "VERWIJZING VORIGE AKTE": Schrappen, tenzij het dossier een basisakte/verkavelingsakte bevat.

Categorie 1b: Juridisch Bepaalde & Optionele Clausules
Logica: BEHOUDEN als de situatie van de partijen het relevant maakt, tenzij het juridisch onmogelijk is.
- Clausule 8 "BEDING VAN AANWAS MET OPTIE":
  * Schrappen indien juridisch onmogelijk/irrelevant: Er is slechts één koper OF de kopers zijn gehuwd.
  * Anders, BEHOUDEN: (bv. bij wettelijk samenwonenden) als relevante optie voor de cliënten.
- Clausule 9 "ONVERDEELDHEID TUSSEN KOPERS":
  * Schrappen indien irrelevant: clausule is niet toepasbaar indien de Kopers NIET zijn gehuwd. 
  * Anders, BEHOUDEN: als standaardregeling of te bespreken optie.
- Clausule 10 "VERKLARING ANTICIPATIEVE INBRENG":
  * Schrappen indien juridisch onmogelijk/irrelevant: Er is slechts één koper OF de kopers zijn reeds gehuwd.
  * Anders, BEHOUDEN: (bv. bij wettelijk/feitelijk samenwonenden) als relevante optie.
- Clausule 11 "VRUCHTGEBRUIK/BLOTE EIGENDOM": Schrappen indien het dossier een aankoop in volle eigendom bevestigt.
- Clausule 13 "GEZINSWONING": Schrappen, enkel als de verkoper niet gehuwd/wettelijk samenwonend is OF als het pand expliciet niet de gezinswoning is.
- Clausule 51 "OVEREENKOMST KOPERS" (Schuldvordering):
  * Schrappen indien irrelevant: Er is slechts één koper.
  * Anders, BEHOUDEN: als relevante optie om een eventuele ongelijke inbreng te regelen.

Categorie 2: Essentiële Clausules (Nooit verwijderen)
Deze clausules (1-7, 15-27, 33-36, 38, 40, 41, 44, 47-50, 52, 54-60, etc.) zijn fundamenteel en worden ALTIJD behouden.

{klantinfo_text}

[Documenten]
{source_content[:5000]}... [beperkt voor context]

[Clausule om te beoordelen]
{clause_text}

OUTPUT FORMAAT

ANALYSE:

CLAUSULE: [Naam van de clausule]
CATEGORIE: [Categorie 1a: Feitelijk / 1b: Optioneel / 2: Essentieel]
TOEGEPASTE LOGICA: [Beschrijf de redeneerstap]
RESULTAAT TOETSING: [Bv. "Geen bewijs gevonden." OF "Geen juridische onmogelijkheid gevonden."]

FINALE BESLISSING: JA (mag volledig verwijderd worden) / NEE (moet behouden blijven)

REDENERING:
[Synthese die leidt tot de finale beslissing]

BEWIJS:
[Citeer het bewijs]

IMPACT VAN DE BESLISSING:
[Leg uit waarom de beslissing de ontwerpakte verbetert]"""

    try:
        response = model.generate_content(check_prompt)
        result_text = response.text
        
        # Extract decision
        decision_match = re.search(r'\*\*FINALE BESLISSING:\*\*\s*JA', result_text, re.IGNORECASE)
        if not decision_match:
            decision_match = re.search(r'FINALE BESLISSING:\s*JA', result_text, re.IGNORECASE)
        
        may_skip = decision_match is not None
        
        return may_skip, result_text
            
    except Exception as e:
        return False, f"Error tijdens analyse: {str(e)}"

def review_agent_check(prompt, research_data, clause_type, model):
    """Review agent that analyzes what's missing based on research"""
    review_prompt = f"""You are a legal review agent. Based on the research findings, determine what additional information is TRULY needed.

IMPORTANT: The research agent has already found information. Only mark something as missing if it was NOT found or had a None/null value.

CLAUSE TYPE: {clause_type}

ORIGINAL PROMPT:
{prompt}

RESEARCH AGENT FINDINGS:
- Research Summary: {research_data.get('research_summary', 'N/A')}
- Found Information: {json.dumps(research_data.get('found_information', {}), ensure_ascii=False)}
- Missing Information: {json.dumps(research_data.get('missing_information', []), ensure_ascii=False)}

CRITICAL INSTRUCTIONS:
1. If research found "repertorium_number: 224455", then repertorium IS NOT MISSING
2. If research found something with value "None" or "null", then it IS missing
3. Check exact values - don't just look at the missing list
4. Only list items that are truly needed and not already found

Respond in JSON format (ALL IN DUTCH):
{{
    "analysis": "korte analyse van wat nodig is",
    "applicable_scenario": "welk scenario van toepassing is",
    "already_found": ["items that research already found with valid values"],
    "critical_missing": ["ONLY items that are truly missing - had None/null values or weren't found"],
    "questions_for_user": [
        {{
            "missing_info": "wat ontbreekt",
            "question": "de vraag aan gebruiker",
            "options": ["optie 1", "optie 2", "optie 3", "optie 4", "Anders"],
            "importance": "CRITICAL/HIGH/MEDIUM"
        }}
    ],
    "can_proceed_without": ["lijst van info die wenselijk maar niet kritiek is"],
    "not_applicable_info": ["lijst van informatie die NIET nodig is"]
}}"""
    
    try:
        response = model.generate_content(review_prompt)
        result_text = response.text
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "analysis": "Review parsing failed",
                "applicable_scenario": "Unknown",
                "already_found": [],
                "critical_missing": [],
                "questions_for_user": [],
                "can_proceed_without": [],
                "not_applicable_info": []
            }
    except Exception as e:
        return {
            "analysis": f"Review error: {str(e)}",
            "applicable_scenario": "Unknown",
            "already_found": [],
            "critical_missing": [],
            "questions_for_user": [],
            "can_proceed_without": [],
            "not_applicable_info": []
        }

def focused_search_for_missing_info(missing_info, source_content, notarial_info, model):
    """Perform a focused search for specific missing information"""
    
    # Format notarial info as searchable text
    notarial_text = "\n\n--- NOTARIËLE INFORMATIE ---\n"
    
    # Add all fields from notarial info
    for key, value in notarial_info.items():
        if key == 'verkopers':
            notarial_text += f"\nVERKOPERS:\n"
            for v in value:
                for k, val in v.items():
                    notarial_text += f"  verkoper_{v['volgnummer']}_{k}: {val}\n"
        elif key == 'kopers':
            notarial_text += f"\nKOPERS:\n"
            for koper in value:
                for k, val in koper.items():
                    notarial_text += f"  koper_{koper['volgnummer']}_{k}: {val}\n"
        elif key == 'user_answers':
            notarial_text += f"\nEERDER BEANTWOORDE VRAGEN:\n"
            for k, val in value.items():
                if isinstance(val, dict):
                    notarial_text += f"  {val.get('missing_info', k)}: {val.get('answer', val)}\n"
        else:
            notarial_text += f"{key}: {value}\n"
    
    search_prompt = f"""You are a specialized legal document search agent. Your task is to find VERY SPECIFIC information.

MISSING INFORMATION TO FIND:
{missing_info}

SEARCH IN TWO PLACES:
1. First check the NOTARIAL INFORMATION (already collected data)
2. Then check the SOURCE DOCUMENTS

--- NOTARIAL INFORMATION TO SEARCH ---
{notarial_text}

--- SOURCE DOCUMENTS TO SEARCH ---
{source_content[:12000]}

CRITICAL CONTEXT FOR NOTARIAL TERMS:
- "day_and_month" or "dag en maand" = the day and month from the ondertekening_datum (signing date)
- "aktedag" = the date of signing the deed = ondertekening_datum
- If looking for day/month and you find ondertekening_datum like "30-10-1990", extract day=30, month=oktober
- Dutch months: januari, februari, maart, april, mei, juni, juli, augustus, september, oktober, november, december
- "repertorium_nummer" = repertory number for the deed
- "Rijksregisternummer" = Belgian national register number (format: YY.MM.DD-XXX.XX)

For electrical installations:
- "Nieuwe installatie" = had complete inspection BEFORE use
- "Oude installatie" = did NOT have inspection before use
- Look for dates in formats: "1 april 2015", "01/04/2015"
- Inspection organizations: BTV, Vinçotte, AIB, ACEG, etc.

INSTRUCTIONS:
1. FIRST check if this information already exists in the notarial information section
2. If not found there, search the source documents
3. Look for variations in spelling, formatting, and phrasing
4. Consider that the same information might be stored under different keys

Return JSON format:
{{
    "search_performed": true,
    "found_in": "notarial_info/source_docs/not_found",
    "found_items": {{
        "{missing_info}": {{
            "found": true/false,
            "value": "exact value found or null",
            "location": "where it was found (notarial info field or document section)",
            "context": "surrounding text or field name where found",
            "confidence": "HIGH/MEDIUM/LOW"
        }}
    }},
    "search_notes": "explanation of search process and any important observations"
}}"""

    try:
        response = model.generate_content(search_prompt)
        result_text = response.text
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return None
    except Exception as e:
        return None

def create_complete_information_set(research_data, user_answers, model):
    """Agent that creates a complete information set from research and user input"""
    
    # Extract key user decisions from answers
    user_decisions = {}
    for key, answer_data in user_answers.items():
        if isinstance(answer_data, dict):
            # Check for negative confirmations
            answer_text = answer_data.get('answer', '').lower()
            question_text = answer_data.get('question', '').lower()
            
            # Detect boolean decisions
            if 'nee' in answer_text or 'geen' in answer_text:
                # Extract what was negated
                if 'beding van aanwas' in question_text:
                    user_decisions['beding_van_aanwas_aanwezig'] = False
                elif 'tontine' in question_text:
                    user_decisions['tontine_aanwezig'] = False
    
    compile_prompt = f"""You are a legal information compiler. Create a complete information set by combining:

RESEARCH FINDINGS:
{json.dumps(research_data.get('found_information', {}), ensure_ascii=False, indent=2)}

USER PROVIDED INFORMATION:
{json.dumps(user_answers, ensure_ascii=False, indent=2)}

USER DECISIONS (extracted):
{json.dumps(user_decisions, ensure_ascii=False, indent=2)}

CRITICAL INSTRUCTION:
- When the user explicitly confirms something is NOT present (e.g., "Nee, geen beding van aanwas"), this OVERRIDES any other information
- Include explicit negative confirmations in the complete_information set
- Flag conditions that should NOT be included in the final clause

Create a comprehensive information set that includes all necessary details for generating a complete legal clause.
Merge research findings with user input, prioritizing user input when there are conflicts.

Respond in JSON format (IN DUTCH):
{{
    "complete_information": {{
        "key": {{
            "value": "waarde",
            "source": "research/user/combined",
            "confidence": "HIGH/MEDIUM/LOW"
        }}
    }},
    "excluded_conditions": ["lijst van condities die NIET toegepast moeten worden"],
    "compilation_notes": "notities over de samenstelling",
    "ready_for_generation": true/false
}}"""

    try:
        response = model.generate_content(compile_prompt)
        result_text = response.text
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {
                "complete_information": {},
                "excluded_conditions": [],
                "compilation_notes": "Compilation failed",
                "ready_for_generation": False
            }
    except Exception as e:
        return {
            "complete_information": {},
            "excluded_conditions": [],
            "compilation_notes": f"Error: {str(e)}",
            "ready_for_generation": False
        }

def generate_final_clause(prompt, complete_info, research_data, source_content, model):
    """Generate the final clause with complete information"""
    # Build comprehensive context
    info_context = "\n\nCOMPLETE INFORMATION SET:\n"
    
    # Include original research findings
    info_context += "\nFROM RESEARCH:\n"
    for key, data in research_data.get('found_information', {}).items():
        info_context += f"- {key}: {data['value']}\n"
    
    # Include compiled information
    info_context += "\nFROM COMPILATION:\n"
    for key, data in complete_info.get('complete_information', {}).items():
        info_context += f"- {key}: {data['value']} (bron: {data['source']})\n"
    
    # Include excluded conditions
    excluded_conditions = complete_info.get('excluded_conditions', [])
    if excluded_conditions:
        info_context += "\nEXCLUDED CONDITIONS (DO NOT GENERATE):\n"
        for condition in excluded_conditions:
            info_context += f"- {condition}\n"
    
    # Get applicable scenario from research or review
    applicable_scenario = research_data.get('applicable_scenario', '')
    research_summary = research_data.get('research_summary', '')
    
    final_prompt = f"""Generate a complete legal clause based on the following:

ORIGINAL PROMPT:
{prompt}

RESEARCH SUMMARY:
{research_summary}

APPLICABLE SCENARIO:
{applicable_scenario}

{info_context}

SOURCE DOCUMENTS:
{source_content}

CRITICAL INSTRUCTIONS:
1. Pay careful attention to the RESEARCH SUMMARY and APPLICABLE SCENARIO
2. ABSOLUTELY DO NOT generate any text related to conditions listed under "EXCLUDED CONDITIONS"
3. Respect conditional logic: if research or user confirms conditions are NOT met, do NOT generate those sections
4. Only generate clause text that matches the actual situation found in the documents
5. Do NOT include block indicators like [BLOCK ...] or [/BLOCK] in the output
6. If a condition is explicitly excluded or false, omit ANY text related to that condition

CONDITIONAL LOGIC RULES:
- When information indicates a condition is "false", "not present", "nee", or listed in EXCLUDED CONDITIONS, exclude ALL related text
- When user explicitly confirms something is NOT present, this overrides any template suggestions
- Generate ONLY text for conditions that are confirmed as true or applicable

Create a legally sound clause using ONLY the applicable information based on the actual conditions found.
If the entire clause depends on an excluded condition, return an appropriate message explaining why the clause cannot be generated.
Ensure the clause is complete, clear, and professionally written in Dutch."""

    response = model.generate_content(final_prompt)
    
    # Clean up the response
    cleaned_text = response.text
    
    # Remove block indicators using regex
    cleaned_text = re.sub(r'\[BLOCK\s+[^\]]+\]', '', cleaned_text)
    cleaned_text = re.sub(r'\[/BLOCK\]', '', cleaned_text)
    
    # Clean up any extra whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

# ============= STREAMLIT UI FUNCTIONS =============

 # Add this at the beginning of your main() function:
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

def main():
    
     """Main function to run the Streamlit app"""
    st.title("🏛️ Notariële Clausule Processor")

# In your main() function:
if not check_password():
    st.stop()  # Do not continue if check_password is not True
    
    st.title("🏛️ Notariële Clausule Processor")
    st.markdown("Automatische verwerking van notariële clausules met AI")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📍 Navigatie")
        
        steps = {
            'intake': '1. 📋 Intake Informatie',
            'documents': '2. 📄 Documenten Uploaden',
            'clauses': '3. 📝 Clausules Verwerken',
            'export': '4. 💾 Exporteren'
        }
        
        for step_key, step_name in steps.items():
            if st.button(step_name, key=f"nav_{step_key}", 
                        type="primary" if st.session_state.current_step == step_key else "secondary"):
                st.session_state.current_step = step_key
                st.rerun()
        
        # Show progress
        st.divider()
        st.subheader("📊 Voortgang")
        
        progress_items = [
            ("Intake compleet", bool(st.session_state.notarial_info)),
            ("Documenten geladen", bool(st.session_state.source_content)),
            ("CSV geüpload", bool(st.session_state.csv_data)),
            (f"Clausules verwerkt ({len(st.session_state.processed_clauses)})", 
             len(st.session_state.processed_clauses) > 0)
        ]
        
        for item, completed in progress_items:
            if completed:
                st.success(f"✅ {item}")
            else:
                st.info(f"⭕ {item}")
    
    # Main content based on current step
    if st.session_state.current_step == 'intake':
        show_intake_form()
    elif st.session_state.current_step == 'documents':
        show_document_upload()
    elif st.session_state.current_step == 'clauses':
        show_clause_processor()
    elif st.session_state.current_step == 'export':
        show_export_section()

def show_intake_form():
    """Show the intake form for notarial information"""
    st.header("📋 Notariële Informatie Verzamelen")
    
    # Check if we can auto-extract
    if st.session_state.source_content:
        if st.button("🤖 Probeer informatie automatisch te extraheren", type="secondary"):
            with st.spinner("Analyseren van documenten..."):
                extracted_data = extract_info_from_documents(st.session_state.source_content)
                if extracted_data:
                    st.success("✅ Informatie geëxtraheerd! Controleer en vul aan waar nodig.")
                    # Pre-fill form with extracted data
                    # This would require more complex state management
    
    with st.form("intake_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏛️ Algemene Informatie")
            
            # Date picker
            signing_date = st.date_input(
                "Datum ondertekening",
                value=datetime.now(),
                format="DD-MM-YYYY"
            )
            
            repertorium = st.text_input("Repertorium nummer")
            
            video_conf = st.checkbox("Akte via videoconferentie?")
        
        with col2:
            st.subheader("📑 Transactie Type")
            
            verkoper_type = st.selectbox(
                "Wie verkoopt?",
                options=[
                    ("alleenstaande", "Een alleenstaande persoon"),
                    ("gehuwd_koppel", "Een gehuwd koppel"),
                    ("wettelijk_samenwonend", "Een wettelijk samenwonend koppel"),
                    ("feitelijk_samenwonend", "Een feitelijk samenwonend koppel"),
                    ("vennootschap", "Een vennootschap")
                ],
                format_func=lambda x: x[1]
            )
            
            koper_type = st.selectbox(
                "Wie koopt?",
                options=[
                    ("alleenstaande", "Een alleenstaande persoon"),
                    ("gehuwd_koppel", "Een gehuwd koppel"),
                    ("wettelijk_samenwonend", "Een wettelijk samenwonend koppel"),
                    ("feitelijk_samenwonend", "Een feitelijk samenwonend koppel"),
                    ("vennootschap", "Een vennootschap")
                ],
                format_func=lambda x: x[1]
            )
        
        st.subheader("💰 Aankoop Details")
        
        col3, col4 = st.columns(2)
        
        with col3:
            aankoop_wijze = st.multiselect(
                "Hoe wordt er gekocht?",
                options=[
                    ("volle_eigendom", "In volle eigendom"),
                    ("gesplitste_aankoop", "Gesplitste aankoop (VG/BE)"),
                    ("gemeenschappelijk_vermogen", "In het gemeenschappelijk vermogen"),
                    ("eigen_onverdeelde_helft", "Elk voor de eigen onverdeelde helft"),
                    ("met_aanwas", "Met beding van aanwas"),
                    ("zonder_aanwas", "Zonder beding van aanwas")
                ],
                format_func=lambda x: x[1]
            )
        
        with col4:
            verkoop_object = st.multiselect(
                "Wat wordt er verkocht?",
                options=[
                    ("enkel_onroerend", "Enkel onroerend goed"),
                    ("met_roerend", "Onroerend + roerend goed"),
                    ("zonder_meetplan", "Zonder recent meetplan"),
                    ("met_meetplan", "Met recent meetplan")
                ],
                format_func=lambda x: x[1]
            )
        
        historiek = st.selectbox(
            "Historiek",
            options=[
                ("zelf_gekocht", "Verkoper heeft zelf gekocht"),
                ("via_schenking", "Verkoper kreeg via schenking"),
                ("ouders_aan_kind", "Verkoop ouders aan kind")
            ],
            format_func=lambda x: x[1]
        )
        
        # Dynamic sections for sellers and buyers
        st.subheader("👤 Verkopers")
        aantal_verkopers = st.number_input("Aantal verkopers", min_value=1, max_value=10, value=1)
        
        verkopers = []
        for i in range(aantal_verkopers):
            with st.expander(f"Verkoper {i+1}"):
                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    voornaam = st.text_input(f"Voornaam", key=f"v_voornaam_{i}")
                    achternaam = st.text_input(f"Achternaam", key=f"v_achternaam_{i}")
                    rijksregister = st.text_input(f"Rijksregisternummer", key=f"v_rr_{i}")
                with vcol2:
                    adres = st.text_area(f"Adres", key=f"v_adres_{i}")
                    burgerlijke_staat = st.selectbox(
                        f"Burgerlijke staat",
                        options=["ongehuwd", "gehuwd", "wettelijk samenwonend", "gescheiden", "weduwe/weduwnaar"],
                        key=f"v_bs_{i}"
                    )
                    aanwezig = st.checkbox(f"Persoonlijk aanwezig?", key=f"v_aanwezig_{i}")
                
                verkopers.append({
                    'volgnummer': i + 1,
                    'voornaam': voornaam,
                    'achternaam': achternaam,
                    'rijksregisternummer': rijksregister,
                    'adres': adres,
                    'burgerlijke_staat': burgerlijke_staat,
                    'aanwezig': aanwezig
                })
        
        st.subheader("👥 Kopers")
        aantal_kopers = st.number_input("Aantal kopers", min_value=1, max_value=10, value=1)
        
        kopers = []
        for i in range(aantal_kopers):
            with st.expander(f"Koper {i+1}"):
                kcol1, kcol2 = st.columns(2)
                with kcol1:
                    voornaam = st.text_input(f"Voornaam", key=f"k_voornaam_{i}")
                    achternaam = st.text_input(f"Achternaam", key=f"k_achternaam_{i}")
                    rijksregister = st.text_input(f"Rijksregisternummer", key=f"k_rr_{i}")
                with kcol2:
                    adres = st.text_area(f"Adres", key=f"k_adres_{i}")
                    burgerlijke_staat = st.selectbox(
                        f"Burgerlijke staat",
                        options=["ongehuwd", "gehuwd", "wettelijk samenwonend", "gescheiden", "weduwe/weduwnaar"],
                        key=f"k_bs_{i}"
                    )
                    aanwezig = st.checkbox(f"Persoonlijk aanwezig?", key=f"k_aanwezig_{i}")
                
                kopers.append({
                    'volgnummer': i + 1,
                    'voornaam': voornaam,
                    'achternaam': achternaam,
                    'rijksregisternummer': rijksregister,
                    'adres': adres,
                    'burgerlijke_staat': burgerlijke_staat,
                    'aanwezig': aanwezig
                })
        
        submitted = st.form_submit_button("💾 Informatie Opslaan", type="primary")
        
        if submitted:
            # Save to session state
            st.session_state.notarial_info = {
                'notary_name': 'Stephane Van Roosbroek',
                'notary_location': '2530 Boechout',
                'notary_office_address': 'Heuvelstraat 54',
                'ondertekening_datum': signing_date.strftime("%d-%m-%Y"),
                'ondertekening_dag': signing_date.day,
                'ondertekening_maand': signing_date.strftime("%B"),
                'ondertekening_maand_nl': get_dutch_month(signing_date.month),
                'ondertekening_jaar': signing_date.year,
                'repertorium_nummer': repertorium,
                'videoconferentie': video_conf,
                'verkoper_type': verkoper_type[0],
                'koper_type': koper_type[0],
                'aankoop_wijze': [x[0] for x in aankoop_wijze],
                'verkoop_object': [x[0] for x in verkoop_object],
                'historiek': historiek[0],
                'verkopers': verkopers,
                'kopers': kopers,
                'verkopers_aanwezig': [{'volgnummer': v['volgnummer'], 'aanwezig': v['aanwezig']} for v in verkopers],
                'kopers_aanwezig': [{'volgnummer': k['volgnummer'], 'aanwezig': k['aanwezig']} for k in kopers],
                'user_answers': {}
            }
            
            st.success("✅ Informatie succesvol opgeslagen!")
            st.balloons()
            
            # Auto navigate to next step
            time.sleep(1)
            st.session_state.current_step = 'documents'
            st.rerun()

def show_document_upload():
    """Show document upload section"""
    st.header("📄 Documenten Uploaden")
    
    st.info("Upload de brondocumenten die gebruikt worden voor de clausule verwerking.")
    
    uploaded_files = st.file_uploader(
        "Kies documenten",
        type=['pdf', 'txt', 'text'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} bestand(en) geüpload")
        
        # Show uploaded files
        for file in uploaded_files:
            st.write(f"📎 {file.name} ({file.size:,} bytes)")
        
        if st.button("🔄 Documenten Verwerken", type="primary"):
            with st.spinner("Documenten worden verwerkt..."):
                source_content = load_source_documents(uploaded_files)
                st.session_state.source_content = source_content
                
                # Add notarial info to source content
                notarial_text = format_notarial_info_as_text(st.session_state.notarial_info)
                st.session_state.source_content += notarial_text
                
                st.success("✅ Documenten succesvol verwerkt!")
                st.info(f"Totale content lengte: {len(st.session_state.source_content):,} karakters")
                
                # Auto navigate to clauses
                time.sleep(1)
                st.session_state.current_step = 'clauses'
                st.rerun()
    
    # Show sample of loaded content if available
    if st.session_state.source_content:
        with st.expander("📋 Voorbeeld van geladen content"):
            st.text(st.session_state.source_content[:1000] + "...")

def show_clause_processor():
    """Show clause processing section with full agent functionality"""
    st.header("📝 Clausules Verwerken")
    
    if not st.session_state.source_content:
        st.warning("⚠️ Upload eerst documenten voordat u clausules kunt verwerken.")
        if st.button("Ga naar Documenten"):
            st.session_state.current_step = 'documents'
            st.rerun()
        return
    
    # Load CSV file
    csv_file = st.file_uploader("Upload clausule CSV bestand", type=['csv'])
    
    if csv_file:
        df = pd.read_csv(csv_file)
        st.session_state.csv_data = df
        
        # Show available clauses
        st.subheader("Beschikbare Clausules")
        
        # Create a more user-friendly display
        clause_options = []
        for i, row in df.iterrows():
            clause_name = row.get('clause', 'Unknown')
            display_name = clause_name.replace('_CLAUSULE', '').replace('_', ' ').title()
            
            # Mark essential clauses
            if (i+1) in ESSENTIAL_CLAUSES:
                display_name += " ⭐ (Essentieel)"
            
            clause_options.append((i+1, display_name))
        
        selected_clause = st.selectbox(
            "Selecteer een clausule om te verwerken",
            options=clause_options,
            format_func=lambda x: f"{x[0]}. {x[1]}"
        )
        
        if st.button("🚀 Start Clausule Verwerking", type="primary"):
            row_number = selected_clause[0]
            clause_name = selected_clause[1]
            
            # Store current processing state
            st.session_state.processing_state = {
                'row_number': row_number,
                'clause_name': clause_name,
                'stage': 'applicability',
                'research_data': None,
                'review_result': None,
                'user_decision': None,
                'questions': [],
                'current_question_index': 0
            }
            st.rerun()
    
    # Handle ongoing processing
    if 'processing_state' in st.session_state and st.session_state.processing_state:
        process_clause_workflow()
    
    # Show processed clauses
    if st.session_state.processed_clauses:
        st.divider()
        st.subheader("🗂️ Verwerkte Clausules")
        for clause_name, content in st.session_state.processed_clauses.items():
            with st.expander(f"📄 {clause_name}"):
                st.text_area("", value=content, height=200, key=f"processed_{clause_name}")

def process_clause_workflow():
    """Handle the multi-stage clause processing workflow"""
    state = st.session_state.processing_state
    
    if not state:
        return
    
    # Get the row data
    row_idx = state['row_number'] - 1
    row = st.session_state.csv_data.iloc[row_idx]
    clause_type = row.get('clause', '')
    prompt = row['optimized_prompt']
    
    # Get skip conditions
    skip_conditions = ""
    if len(row) > 3:
        skip_conditions = row.iloc[3] if pd.notna(row.iloc[3]) else ""
    elif 'skip_conditions' in row:
        skip_conditions = row.get('skip_conditions', '')
    
    # Initialize model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Stage 1: Applicability Check
    if state['stage'] == 'applicability':
        st.info(f"🤖 Verwerking van: **{state['clause_name']}**")
        
        is_essential_clause = state['row_number'] in ESSENTIAL_CLAUSES
        
        if is_essential_clause:
            st.success("✅ ESSENTIËLE CLAUSULE - wordt altijd toegepast")
            state['user_decision'] = 'apply'
            state['stage'] = 'research'
            st.rerun()
        else:
            with st.spinner("⚖️ Controleren of clausule van toepassing is..."):
                may_skip, analysis = check_clause_applicability(
                    prompt, clause_type, skip_conditions, 
                    st.session_state.source_content, 
                    st.session_state.notarial_info, 
                    model
                )
            
            # Show analysis
            with st.expander("📋 Applicability Analyse", expanded=True):
                st.text(analysis)
            
            st.warning(f"De AI adviseert: {'Clausule MAG verwijderd worden' if may_skip else 'Clausule MOET behouden blijven'}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Clausule Toepassen", type="primary"):
                    state['user_decision'] = 'apply'
                    state['stage'] = 'research'
                    st.rerun()
            with col2:
                if st.button("❌ Clausule Overslaan", type="secondary"):
                    state['user_decision'] = 'skip'
                    state['stage'] = 'complete'
                    st.rerun()
    
    # Stage 2: Research
    elif state['stage'] == 'research':
        with st.spinner("🔬 Research Agent analyseert informatie behoeften..."):
            research_data = research_agent_determine_needs(
                prompt, clause_type, st.session_state.source_content, model
            )
            state['research_data'] = research_data
        
        # Show research results
        st.success("✅ Research compleet!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gevonden informatie", len(research_data.get('found_information', {})))
        with col2:
            st.metric("Ontbrekende informatie", len(research_data.get('missing_information', [])))
        
        with st.expander("🔍 Research Details"):
            st.write("**Samenvatting:**", research_data.get('research_summary', 'N/A'))
            st.write("**Applicable Scenario:**", research_data.get('applicable_scenario', 'N/A'))
            
            if research_data.get('found_information'):
                st.write("**Gevonden informatie:**")
                for key, info in research_data['found_information'].items():
                    st.write(f"- {key}: {info['value']} (confidence: {info['confidence']})")
        
        state['stage'] = 'review'
        st.rerun()
    
    # Stage 3: Review
    elif state['stage'] == 'review':
        with st.spinner("📋 Review Agent bepaalt wat nog nodig is..."):
            review_result = review_agent_check(
                prompt, state['research_data'], clause_type, model
            )
            state['review_result'] = review_result
        
        st.info("📊 Review Analyse compleet")
        
        if review_result.get('critical_missing'):
            st.warning(f"❓ Er zijn {len(review_result['critical_missing'])} vragen voor u")
            state['questions'] = review_result.get('questions_for_user', [])
            state['current_question_index'] = 0
            state['stage'] = 'questions'
        else:
            st.success("✅ Alle benodigde informatie is beschikbaar!")
            state['stage'] = 'generation'
        
        st.rerun()
    
    # Stage 4: Questions
    elif state['stage'] == 'questions':
        if state['current_question_index'] < len(state['questions']):
            current_q = state['questions'][state['current_question_index']]
            
            st.subheader(f"❓ Vraag {state['current_question_index'] + 1} van {len(state['questions'])}")
            
            # Try focused search first
            with st.spinner(f"🔍 Zoeken naar: {current_q['missing_info']}..."):
                focused_result = focused_search_for_missing_info(
                    current_q['missing_info'], 
                    st.session_state.source_content,
                    st.session_state.notarial_info,
                    model
                )
            
            found_automatically = False
            if focused_result and focused_result.get('found_items'):
                for item_key, item_data in focused_result['found_items'].items():
                    if item_data.get('found') and item_data.get('value'):
                        st.success(f"✅ Automatisch gevonden: {item_data['value']}")
                        
                        # Store the answer
                        answer_key = f"{clause_type}_{current_q['missing_info']}"
                        if 'user_answers' not in st.session_state.notarial_info:
                            st.session_state.notarial_info['user_answers'] = {}
                        
                        st.session_state.notarial_info['user_answers'][answer_key] = {
                            "question": current_q['question'],
                            "answer": str(item_data['value']),
                            "missing_info": current_q['missing_info'],
                            "clause_type": clause_type,
                            "source": "focused_search",
                            "confidence": item_data.get('confidence', 'HIGH')
                        }
                        
                        found_automatically = True
                        state['current_question_index'] += 1
                        st.rerun()
            
            if not found_automatically:
                # Ask user
                st.warning(f"**{current_q['missing_info']}**")
                st.write(current_q['question'])
                
                options = current_q.get('options', [])
                if options:
                    selected_option = st.radio("Selecteer een optie:", options, key=f"q_{state['current_question_index']}")
                    
                    if "anders" in selected_option.lower():
                        custom_answer = st.text_input("Specificeer:", key=f"custom_{state['current_question_index']}")
                        if custom_answer:
                            selected_option = custom_answer
                    
                    if st.button("➡️ Volgende", type="primary"):
                        # Store answer
                        answer_key = f"{clause_type}_{current_q['missing_info']}"
                        if 'user_answers' not in st.session_state.notarial_info:
                            st.session_state.notarial_info['user_answers'] = {}
                        
                        st.session_state.notarial_info['user_answers'][answer_key] = {
                            "question": current_q['question'],
                            "answer": selected_option,
                            "missing_info": current_q['missing_info'],
                            "clause_type": clause_type
                        }
                        
                        state['current_question_index'] += 1
                        st.rerun()
                else:
                    answer = st.text_input("Uw antwoord:", key=f"text_{state['current_question_index']}")
                    if st.button("➡️ Volgende", type="primary") and answer:
                        # Store answer
                        answer_key = f"{clause_type}_{current_q['missing_info']}"
                        if 'user_answers' not in st.session_state.notarial_info:
                            st.session_state.notarial_info['user_answers'] = {}
                        
                        st.session_state.notarial_info['user_answers'][answer_key] = {
                            "question": current_q['question'],
                            "answer": answer,
                            "missing_info": current_q['missing_info'],
                            "clause_type": clause_type
                        }
                        
                        state['current_question_index'] += 1
                        st.rerun()
        else:
            # All questions answered
            state['stage'] = 'generation'
            st.rerun()
    
    # Stage 5: Generation
    elif state['stage'] == 'generation':
        st.info("📝 Genereren van clausule...")
        
        # Get user answers for this clause
        clause_user_answers = {}
        if 'user_answers' in st.session_state.notarial_info:
            for key, value in st.session_state.notarial_info['user_answers'].items():
                if key.startswith(f"{clause_type}_"):
                    clause_user_answers[key] = value
        
        # Create complete information set
        with st.spinner("🔧 Compileren van informatie..."):
            if clause_user_answers:
                complete_info = create_complete_information_set(
                    state['research_data'], 
                    clause_user_answers, 
                    model
                )
            else:
                # No user answers needed
                complete_info = {
                    "complete_information": {},
                    "excluded_conditions": [],
                    "compilation_notes": "All info from research",
                    "ready_for_generation": True
                }
                # Convert research data
                for key, data in state['research_data'].get('found_information', {}).items():
                    if isinstance(data, dict) and 'value' in data:
                        complete_info["complete_information"][key] = {
                            "value": data['value'],
                            "source": "research",
                            "confidence": data.get('confidence', 'HIGH')
                        }
        
        # Generate final clause
        with st.spinner("✍️ Genereren van finale clausule..."):
            final_clause = generate_final_clause(
                prompt, 
                complete_info, 
                state['research_data'], 
                st.session_state.source_content, 
                model
            )
        
        st.success("✅ Clausule succesvol gegenereerd!")
        
        # Store the result
        st.session_state.processed_clauses[state['clause_name']] = final_clause
        
        # Show the result
        st.text_area("Gegenereerde Clausule:", value=final_clause, height=300)
        
        state['stage'] = 'complete'
        st.rerun()
    
    # Stage 6: Complete
    elif state['stage'] == 'complete':
        if state['user_decision'] == 'skip':
            st.info("⏭️ Clausule overgeslagen op gebruikersbeslissing")
        else:
            st.success(f"✅ Clausule '{state['clause_name']}' verwerkt en opgeslagen!")
        
        # Clear processing state
        st.session_state.processing_state = {}
        
        if st.button("🔄 Volgende Clausule Verwerken"):
            st.rerun()

def show_export_section():
    """Show export section with multiple export options"""
    st.header("💾 Exporteren")
    
    if not st.session_state.processed_clauses:
        st.warning("⚠️ Er zijn nog geen clausules verwerkt om te exporteren.")
        if st.button("Ga naar Clausules"):
            st.session_state.current_step = 'clauses'
            st.rerun()
        return
    
    st.success(f"✅ {len(st.session_state.processed_clauses)} clausule(s) klaar voor export")
    
    # Show summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Samenvatting")
        st.metric("Verwerkte Clausules", len(st.session_state.processed_clauses))
        st.metric("Aantal Verkopers", len(st.session_state.notarial_info.get('verkopers', [])))
        st.metric("Aantal Kopers", len(st.session_state.notarial_info.get('kopers', [])))
    
    with col2:
        st.subheader("📋 Transactie Details")
        st.write(f"**Verkoper type:** {st.session_state.notarial_info.get('verkoper_type', 'N/A').replace('_', ' ')}")
        st.write(f"**Koper type:** {st.session_state.notarial_info.get('koper_type', 'N/A').replace('_', ' ')}")
        st.write(f"**Datum:** {st.session_state.notarial_info.get('ondertekening_datum', 'N/A')}")
        st.write(f"**Repertorium:** {st.session_state.notarial_info.get('repertorium_nummer', 'N/A')}")
    
    st.divider()
    
    # Export options
    st.subheader("📤 Export Opties")
    
    # Prepare export content
    export_content = "NOTARIËLE AKTE\n" + "="*50 + "\n\n"
    
    # Add header information
    export_content += "ALGEMENE INFORMATIE\n" + "-"*30 + "\n"
    export_content += f"Datum: {st.session_state.notarial_info.get('ondertekening_datum', 'N/A')}\n"
    export_content += f"Repertorium nummer: {st.session_state.notarial_info.get('repertorium_nummer', 'N/A')}\n"
    export_content += f"Notaris: {st.session_state.notarial_info.get('notary_name', 'N/A')}\n"
    export_content += f"Kantoor: {st.session_state.notarial_info.get('notary_office_address', 'N/A')}, "
    export_content += f"{st.session_state.notarial_info.get('notary_location', 'N/A')}\n\n"
    
    # Add clauses
    export_content += "CLAUSULES\n" + "="*50 + "\n\n"
    
    for i, (clause_name, clause_content) in enumerate(st.session_state.processed_clauses.items(), 1):
        export_content += f"{i}. {clause_name}\n" + "-"*30 + "\n"
        export_content += clause_content + "\n\n"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Text export
        st.download_button(
            label="📝 Download als Tekst",
            data=export_content,
            file_name=f"notariele_akte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            type="primary"
        )
    
    with col2:
        # JSON export (structured data)
        export_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "notarial_info": st.session_state.notarial_info,
                "clause_count": len(st.session_state.processed_clauses)
            },
            "clauses": st.session_state.processed_clauses
        }
        
        st.download_button(
            label="🔧 Download als JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"notariele_akte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            type="secondary"
        )
    
    with col3:
        # HTML export
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Notariële Akte - {st.session_state.notarial_info.get('ondertekening_datum', 'N/A')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .clause {{ margin-bottom: 30px; page-break-inside: avoid; }}
        .metadata {{ background-color: #f5f5f5; padding: 15px; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <h1>NOTARIËLE AKTE</h1>
    
    <div class="metadata">
        <h2>Algemene Informatie</h2>
        <p><strong>Datum:</strong> {st.session_state.notarial_info.get('ondertekening_datum', 'N/A')}</p>
        <p><strong>Repertorium nummer:</strong> {st.session_state.notarial_info.get('repertorium_nummer', 'N/A')}</p>
        <p><strong>Notaris:</strong> {st.session_state.notarial_info.get('notary_name', 'N/A')}</p>
        <p><strong>Kantoor:</strong> {st.session_state.notarial_info.get('notary_office_address', 'N/A')}, {st.session_state.notarial_info.get('notary_location', 'N/A')}</p>
    </div>
    
    <h2>Clausules</h2>
    """
        
        for i, (clause_name, clause_content) in enumerate(st.session_state.processed_clauses.items(), 1):
            html_content += f"""
    <div class="clause">
        <h3>{i}. {clause_name}</h3>
        <p>{clause_content.replace('\n', '<br>')}</p>
    </div>
    """
        
        html_content += """
</body>
</html>
"""
        
        st.download_button(
            label="🌐 Download als HTML",
            data=html_content,
            file_name=f"notariele_akte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            type="secondary"
        )
    
    # Preview section
    st.divider()
    st.subheader("👁️ Voorbeeld Export")
    
    with st.expander("Bekijk volledige akte", expanded=False):
        st.text(export_content)
    
    # Options to process more or finish
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Meer Clausules Verwerken", type="primary"):
            st.session_state.current_step = 'clauses'
            st.rerun()
    
    with col2:
        if st.button("🏁 Afsluiten", type="secondary"):
            st.success("✅ Bedankt voor het gebruik van de Notariële Clausule Processor!")
            st.balloons()

if __name__ == "__main__":
    main()