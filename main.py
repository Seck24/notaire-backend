"""
Backend API — Notaire Agentia / Préo IA
FastAPI server pour la génération d'actes notariaux via RAG + Claude
"""

import os
import io
import time
import base64
import logging
import httpx
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ─── CONFIG ───────────────────────────────────────────────────
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "https://rbujxzyvsftvzyxfifke.supabase.co")
SUPABASE_KEY      = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
MISTRAL_KEY       = os.environ["MISTRAL_API_KEY"]
ANTHROPIC_KEY     = os.environ["ANTHROPIC_API_KEY"]
ALLOWED_ORIGINS   = os.environ.get("ALLOWED_ORIGINS", "https://notaire-agentia.preo-ia.info").split(",")
ENVIRONMENT       = os.environ.get("ENVIRONMENT", "production")

TENANT_ID         = "commun"
EMBED_MODEL       = "mistral-embed"
CLAUDE_MODEL      = "claude-sonnet-4-6"
TOP_K             = 8   # chunks RAG à récupérer
WORD_API_URL      = os.environ.get("WORD_API_URL", "http://161.97.181.171:8001")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── CLIENTS ──────────────────────────────────────────────────
from supabase import create_client
from mistralai import Mistral
import anthropic

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
mistral  = Mistral(api_key=MISTRAL_KEY)
claude   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ─── APP ──────────────────────────────────────────────────────
app = FastAPI(
    title="Notaire Agentia API",
    description="API de génération d'actes notariaux — Préo IA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODÈLES ──────────────────────────────────────────────────
class TokenRequest(BaseModel):
    token: str
    cabinet_id: str

class TokenResponse(BaseModel):
    valid: bool
    cabinet_id: Optional[str] = None
    message: str

# ─── UTILITAIRES ──────────────────────────────────────────────

def lire_pdf(contenu: bytes) -> str:
    """Extrait le texte d'un PDF."""
    try:
        import pdfplumber
        texte = []
        with pdfplumber.open(io.BytesIO(contenu)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texte.append(t)
        return "\n".join(texte)
    except Exception as e:
        logger.warning(f"PDF non lisible : {e}")
        return ""


def obtenir_embedding(texte: str) -> list:
    """Embedding Mistral pour une requête."""
    try:
        response = mistral.embeddings.create(
            model=EMBED_MODEL,
            inputs=[texte],
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Erreur embedding Mistral : {e}")
        return []


def rechercher_rag(query: str, type_acte: Optional[str] = None) -> list[dict]:
    """Recherche vectorielle dans Supabase RAG."""
    embedding = obtenir_embedding(query)
    if not embedding:
        return []

    try:
        params = {
            "query_embedding": embedding,
            "match_count": TOP_K,
            "tenant_id_filter": TENANT_ID,
        }
        if type_acte:
            params["type_acte_filter"] = type_acte

        result = supabase.rpc("match_documents_rag", params).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"Erreur RAG Supabase : {e}")
        return []


def construire_prompt(
    type_acte: str,
    contexte_docs: str,
    chunks_rag: list[dict],
    infos_supplementaires: str = "",
) -> str:
    """Construit le prompt Claude pour la génération d'acte."""
    rag_context = "\n\n---\n\n".join(
        f"[Source: {c.get('fichier_source', 'inconnu')}]\n{c.get('contenu', '')}"
        for c in chunks_rag[:TOP_K]
    )

    type_labels = {
        "vente":      "Acte de vente immobilière",
        "societe":    "Constitution de société (SARL/SAS)",
        "succession": "Déclaration de succession",
        "donation":   "Acte de donation",
        "credit":     "Acte d'ouverture de crédit hypothécaire",
        "bail":       "Bail notarié",
    }
    label = type_labels.get(type_acte, type_acte)

    return f"""Tu es un notaire expert en droit ivoirien (CI).
Tu dois rédiger un {label} professionnel et conforme au droit CI.

BASE LÉGALE ET MODÈLES (extraits RAG) :
{rag_context if rag_context else "Aucun document RAG disponible — utiliser les règles générales du droit CI."}

DOCUMENTS FOURNIS PAR LE CLIENT :
{contexte_docs if contexte_docs else "Aucun document fourni."}

INFORMATIONS COMPLÉMENTAIRES :
{infos_supplementaires if infos_supplementaires else "Aucune."}

INSTRUCTIONS :
1. Rédige l'acte complet en français juridique formel
2. Utilise [PLACEHOLDER] pour les données à compléter (noms, dates, montants, etc.)
3. Inclus toutes les clauses obligatoires selon le droit CI
4. Respecte le formalisme notarial ivoirien (formules d'en-tête, de clôture, etc.)
5. Ajoute les références légales (articles de loi, décrets) pertinentes
6. L'acte doit être immédiatement utilisable par un notaire après remplissage des placeholders

Rédige maintenant l'acte complet :"""


# ─── ENDPOINTS ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "environment": ENVIRONMENT, "version": "1.0.0"}


@app.post("/api/verify-token", response_model=TokenResponse)
async def verify_token(request: TokenRequest):
    """Vérifie un token d'accès cabinet."""
    try:
        result = supabase.table("cabinets_tokens") \
            .select("cabinet_id, actif") \
            .eq("token_api", request.token) \
            .eq("cabinet_id", request.cabinet_id) \
            .eq("actif", True) \
            .limit(1) \
            .execute()

        if result.data:
            return TokenResponse(
                valid=True,
                cabinet_id=result.data[0]["cabinet_id"],
                message="Token valide"
            )
        return TokenResponse(valid=False, message="Token invalide ou inactif")
    except Exception as e:
        logger.error(f"Erreur verify-token : {e}")
        return TokenResponse(valid=False, message="Erreur de vérification")


@app.post("/api/generer-acte")
async def generer_acte(
    type_acte: str = Form(...),
    infos: str = Form(default=""),
    fichiers: list[UploadFile] = File(default=[]),
):
    """
    Génère un acte notarial via RAG + Claude.

    - type_acte : vente | societe | succession | donation | credit | bail
    - infos : informations textuelles complémentaires
    - fichiers : documents PDF/TXT uploadés par le client
    """
    # Extraction texte des fichiers uploadés
    textes_docs = []
    for fichier in fichiers:
        contenu = await fichier.read()
        nom = fichier.filename or ""
        if nom.lower().endswith(".pdf"):
            texte = lire_pdf(contenu)
        elif nom.lower().endswith((".txt", ".md")):
            try:
                texte = contenu.decode("utf-8")
            except Exception:
                texte = contenu.decode("latin-1", errors="ignore")
        else:
            texte = ""  # format non supporté

        if texte.strip():
            textes_docs.append(f"=== {nom} ===\n{texte[:3000]}")  # limiter par fichier

    contexte_docs = "\n\n".join(textes_docs)

    # Mapping type_acte → type_acte RAG
    rag_type_acte_map = {
        "vente":      "vente_immobiliere",
        "societe":    "constitution_societe",
        "succession": "succession",
        "donation":   "donation",
        "credit":     "ouverture_credit",
        "bail":       None,  # pas encore dans le RAG
    }
    rag_type_acte = rag_type_acte_map.get(type_acte)

    # Requête RAG
    query = f"acte notarial {type_acte} Côte d'Ivoire droit ivoirien clauses obligatoires"
    chunks_rag = rechercher_rag(query, rag_type_acte)
    logger.info(f"RAG : {len(chunks_rag)} chunks trouvés pour type_acte={type_acte}")

    # Génération Claude
    prompt = construire_prompt(type_acte, contexte_docs, chunks_rag, infos)

    try:
        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        acte_genere = message.content[0].text
    except Exception as e:
        logger.error(f"Erreur Claude : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la génération par l'IA")

    return JSONResponse({
        "success": True,
        "type_acte": type_acte,
        "acte": acte_genere,
        "rag_chunks_utilises": len(chunks_rag),
        "docs_fournis": len(fichiers),
    })


class WordRequest(BaseModel):
    type_acte: str
    texte_acte: str
    cabinet_nom: str = "Étude Notariale"


@app.post("/api/generate-word")
async def generate_word(payload: WordRequest):
    """
    Proxy vers le Word API (port 8001).
    Retourne le fichier .docx en téléchargement direct.
    """
    # Mapping type_acte frontend → Word API
    type_map = {
        "vente":      "vente_immobiliere",
        "societe":    "constitution_societe",
        "succession": "succession",
        "donation":   "donation",
        "credit":     "ouverture_credit",
        "bail":       "vente_immobiliere",  # fallback
    }
    type_acte_word = type_map.get(payload.type_acte, payload.type_acte)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{WORD_API_URL}/generate-word",
                json={
                    "type_acte": type_acte_word,
                    "texte_acte": payload.texte_acte,
                    "cabinet_nom": payload.cabinet_nom,
                },
            )
            response.raise_for_status()
            data = response.json()

        docx_bytes = base64.b64decode(data["docx_base64"])
        filename = data.get("filename", f"acte_{type_acte_word}.docx")

        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except httpx.HTTPError as e:
        logger.error(f"Erreur Word API : {e}")
        raise HTTPException(status_code=502, detail="Erreur lors de la génération Word")
