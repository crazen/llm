"""
main.py — FastAPI backend completo
Features: Streaming, Auto-rename, Export, Rate Limiting, Segurança completa
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional
import os, time, uuid, pickle, json, asyncio
from pathlib import Path
from dotenv import load_dotenv

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# LangChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

import networkx as nx

try:
    import spacy
    nlp_spacy = spacy.load("pt_core_news_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False

from supabase import create_client, Client

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Configuração do ambiente
# ─────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000").split(",")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Configure SUPABASE_URL e SUPABASE_KEY no .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────────────────────────
# App + Rate Limiting
# ─────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# Em produção: docs_url=None, redoc_url=None
IS_PROD = os.getenv("ENV", "dev") == "prod"
app = FastAPI(
    title="NIM Chat API",
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ─────────────────────────────────────────────────────────────────
# Modelos LangChain
# ─────────────────────────────────────────────────────────────────
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
emb = NVIDIAEmbeddings()

memory_cache: dict = {}
vectors_cache: dict = {}
graph_cache: dict = {}

# ─────────────────────────────────────────────────────────────────
# Helpers de disco (com UUID nos uploads para segurança)
# ─────────────────────────────────────────────────────────────────
def user_storage_dir(user_id: str) -> Path:
    base = Path("user_data") / user_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "uploads").mkdir(exist_ok=True)
    return base

def faiss_path(user_id: str, session_id: str) -> Path:
    return user_storage_dir(user_id) / f"faiss_{session_id}"

def graph_path_file(user_id: str, session_id: str) -> Path:
    return user_storage_dir(user_id) / f"graph_{session_id}.pkl"

def uploads_index_path(user_id: str) -> Path:
    """Índice JSON que mapeia nome_original → nome_seguro no disco"""
    return user_storage_dir(user_id) / "uploads_index.json"

def load_uploads_index(user_id: str) -> dict:
    p = uploads_index_path(user_id)
    if p.exists():
        return json.loads(p.read_text())
    return {}

def save_uploads_index(user_id: str, index: dict):
    uploads_index_path(user_id).write_text(json.dumps(index, ensure_ascii=False))

# ─────────────────────────────────────────────────────────────────
# Autenticação
# ─────────────────────────────────────────────────────────────────
async def get_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token não fornecido")
    token = authorization.split(" ", 1)[1]
    try:
        user = supabase.auth.get_user(token)
        return user.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")

# ─────────────────────────────────────────────────────────────────
# Funções de sessão
# ─────────────────────────────────────────────────────────────────
def db_get_sessions(user_id: str):
    try:
        r = supabase.table("chat_sessions").select("*").eq("user_id", user_id).order("created_at").execute()
        return r.data or []
    except Exception:
        return []

def db_create_session(user_id: str, name: str = "Nova Conversa") -> str:
    try:
        r = supabase.table("chat_sessions").insert({"user_id": user_id, "name": name}).execute()
        return r.data[0]["id"]
    except Exception:
        sid = str(uuid.uuid4())
        supabase.table("chat_sessions").insert({"id": sid, "user_id": user_id, "name": name}).execute()
        return sid

def db_rename_session(session_id: str, name: str):
    supabase.table("chat_sessions").update({"name": name}).eq("id", session_id).execute()

def db_delete_session(session_id: str):
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()

def db_load_messages(session_id: str):
    try:
        r = supabase.table("chat_messages").select("*").eq("session_id", session_id).order("created_at").execute()
        return r.data or []
    except Exception:
        return []

def db_save_message(session_id: str, role: str, content: str):
    supabase.table("chat_messages").insert({
        "session_id": session_id, "role": role, "content": content
    }).execute()

# ─────────────────────────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────────────────────────
def load_documents(user_id: str):
    docs = []
    index = load_uploads_index(user_id)
    safe_to_original = {v: k for k, v in index.items()}

    for folder in [Path("docs"), user_storage_dir(user_id) / "uploads"]:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if not f.is_file():
                continue
            try:
                if f.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(f))
                elif f.suffix.lower() == ".txt":
                    loader = TextLoader(str(f), encoding="utf-8")
                else:
                    continue
                loaded = loader.load()
                display_name = safe_to_original.get(f.name, f.name)
                for d in loaded:
                    d.metadata["filename"] = display_name
                docs.extend(loaded)
            except Exception:
                pass
    return docs

def build_or_load_vectors(user_id: str, session_id: str, force: bool = False):
    cache_key = (user_id, session_id)
    if not force and cache_key in vectors_cache:
        return vectors_cache[cache_key]
    fp = faiss_path(user_id, session_id)
    docs = load_documents(user_id)
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    if not force and fp.exists() and not should_rebuild(user_id, session_id):
        try:
            v = FAISS.load_local(str(fp), emb, allow_dangerous_deserialization=True)
            vectors_cache[cache_key] = v
            return v
        except Exception:
            pass
    v = FAISS.from_documents(chunks, emb)
    v.save_local(str(fp))
    vectors_cache[cache_key] = v
    return v

def should_rebuild(user_id: str, session_id: str) -> bool:
    """Verifica se há arquivos mais novos que o índice existente"""
    fp = faiss_path(user_id, session_id)
    if not fp.exists():
        return True
    
    index_mtime = fp.stat().st_mtime
    
    # Verificar pasta docs global
    for folder in [Path("docs"), user_storage_dir(user_id) / "uploads"]:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.is_file() and f.stat().st_mtime > index_mtime:
                return True  # arquivo mais novo que o índice
    
    return False

def extract_entities(text: str):
    if not SPACY_AVAILABLE:
        return []
    try:
        doc = nlp_spacy(text[:50000])
        return [(e.text.strip(), e.label_) for e in doc.ents if len(e.text.strip()) > 1]
    except Exception:
        return []

def build_graph(user_id: str, session_id: str, force: bool = False) -> nx.DiGraph:
    cache_key = (user_id, session_id)
    if not force and cache_key in graph_cache:
        return graph_cache[cache_key]
    gp = graph_path_file(user_id, session_id)
    if not force and gp.exists():
        try:
            with open(gp, "rb") as f:
                g = pickle.load(f)
            graph_cache[cache_key] = g
            return g
        except Exception:
            pass
    docs = load_documents(user_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    G = nx.DiGraph()
    for doc in chunks:
        entities = extract_entities(doc.page_content)
        for et, el in entities:
            if not G.has_node(et):
                G.add_node(et, type=el)
        for i, (e1, _) in enumerate(entities):
            for e2, _ in entities[i+1:i+6]:
                if e1 == e2:
                    continue
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1, context=doc.page_content[:200])
    with open(gp, "wb") as f:
        pickle.dump(G, f)
    graph_cache[cache_key] = G
    return G

def query_graph(G: nx.DiGraph, query: str) -> str:
    if G.number_of_nodes() == 0:
        return ""
    entities = extract_entities(query)
    if not entities:
        words = set(query.lower().split())
        entities = [(n, "") for n in G.nodes() if any(w in n.lower() for w in words if len(w) > 3)]
    if not entities:
        return ""
    relations = []
    visited = set()
    for et, _ in entities[:5]:
        matched = [n for n in G.nodes() if et.lower() in n.lower()]
        for node in matched[:2]:
            for neighbor in list(G.successors(node))[:5]:
                key = f"{node}→{neighbor}"
                if key not in visited:
                    visited.add(key)
                    edge = G[node][neighbor]
                    relations.append(f"• {node} → {neighbor} | {edge.get('context','')[:100]}")
    if not relations:
        return ""
    return "Relações (Knowledge Graph):\n" + "\n".join(relations[:8])

def rerank(query: str, docs: list, top_n: int = 4) -> list:
    if not RERANKER_AVAILABLE or not docs:
        return docs[:top_n]
    try:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker_model.predict(pairs)
        ranked = sorted(zip(scores, docs), reverse=True)
        return [d for _, d in ranked[:top_n]]
    except Exception:
        return docs[:top_n]

def hyde_query(query: str) -> str:
    try:
        return llm.invoke(
            f"Escreva um parágrafo de 3 frases que responderia: {query}\nApenas o parágrafo:"
        ).content.strip()
    except Exception:
        return query

def auto_title(first_message: str) -> str:
    """Gera título curto para a sessão com base na primeira mensagem"""
    try:
        title = llm.invoke(
            f"Crie um título curto (máximo 5 palavras, sem aspas) para uma conversa que começa com:\n{first_message}"
        ).content.strip().strip('"').strip("'")
        return title[:60]
    except Exception:
        return first_message[:40]

def build_prompt_and_retrieve(user_input: str, user_id: str, session_id: str, body):
    """Executa o pipeline RAG completo e retorna (filled_messages, sources, graph_ctx)"""
    vectors = build_or_load_vectors(user_id, session_id)
    graph = build_graph(user_id, session_id) if body.use_graph else nx.DiGraph()

    if not vectors:
        return None, [], ""

    search_q = hyde_query(user_input) if body.use_hyde else user_input

    if body.use_multi_query:
        try:
            variations_prompt = f"""Gere 3 variações diferentes desta pergunta para busca em documentos.
Retorne apenas as 3 perguntas, uma por linha, sem numeração:
Pergunta original: {search_q}"""
            variations_text = llm.invoke(variations_prompt).content.strip()
            queries = [search_q] + [v.strip() for v in variations_text.split("\n") if v.strip()][:3]
            all_docs, seen = [], set()
            for q in queries:
                for doc in vectors.similarity_search(q, k=4):
                    key = doc.page_content[:80]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(doc)
            candidate_docs = all_docs
        except Exception:
            candidate_docs = vectors.similarity_search(search_q, k=8)
    else:
        candidate_docs = vectors.similarity_search(search_q, k=8)

    final_docs = rerank(user_input, candidate_docs, top_n=4) if body.use_reranking else candidate_docs[:4]
    graph_ctx = query_graph(graph, user_input) if body.use_graph else ""

    ctx_text = "\n\n".join(
        [f"[{d.metadata.get('filename','?')}]\n{d.page_content}" for d in final_docs]
    )

    prompt_template = ChatPromptTemplate.from_template("""
Você é um assistente útil e conciso. Use o contexto fornecido para responder.
Regras:
- Responda de forma direta e natural
- Se não encontrar nos documentos, diga: "Não encontrei essa informação nos documentos."
- Seja objetivo e evite repetições

Contexto dos documentos:
{context}

{graph_context}

Pergunta: {input}
Resposta:""")

    filled = prompt_template.format_messages(
        context=ctx_text,
        graph_context=graph_ctx,
        input=user_input,
    )

    sources = [
        {"filename": d.metadata.get("filename", "?"), "preview": d.page_content[:200]}
        for d in final_docs
    ]
    return filled, sources, graph_ctx

# ─────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────
class SessionCreate(BaseModel):
    name: str = "Nova Conversa"

class SessionRename(BaseModel):
    name: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_hyde: bool = True
    use_multi_query: bool = True
    use_reranking: bool = True
    use_graph: bool = True

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None

# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

# ── Sessões ──────────────────────────────────────────────────────
@app.get("/sessions")
async def get_sessions(user_id: str = Depends(get_user_id)):
    return db_get_sessions(user_id)

@app.post("/sessions")
async def create_session(body: SessionCreate, user_id: str = Depends(get_user_id)):
    sid = db_create_session(user_id, body.name)
    return {"id": sid, "name": body.name}

@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: SessionRename, user_id: str = Depends(get_user_id)):
    db_rename_session(session_id, body.name)
    return {"ok": True}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_user_id)):
    db_delete_session(session_id)
    vectors_cache.pop((user_id, session_id), None)
    graph_cache.pop((user_id, session_id), None)
    return {"ok": True}

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, user_id: str = Depends(get_user_id)):
    return db_load_messages(session_id)

# ── Chat normal (JSON) ────────────────────────────────────────────
@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest, user_id: str = Depends(get_user_id)):
    db_save_message(body.session_id, "user", body.message)

    # Auto-rename: se for a primeira mensagem, gera título automático
    msgs = db_load_messages(body.session_id)
    if len(msgs) == 1:
        title = auto_title(body.message)
        db_rename_session(body.session_id, title)

    filled, sources, graph_ctx = build_prompt_and_retrieve(
        body.message, user_id, body.session_id, body
    )

    if filled:
        answer = llm.invoke(filled).content
    else:
        mem_key = f"{user_id}_{body.session_id}"
        if mem_key not in memory_cache:
            memory_cache[mem_key] = []
        memory_cache[mem_key].append(HumanMessage(content=body.message))
        answer = llm.invoke(memory_cache[mem_key]).content
        memory_cache[mem_key].append(AIMessage(content=answer))

    db_save_message(body.session_id, "assistant", answer)
    return {
        "answer": answer,
        "sources": sources,
        "graph_context": graph_ctx,
        "session_renamed": len(msgs) == 1,
    }

# ── Chat streaming (SSE) ──────────────────────────────────────────
@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatRequest, user_id: str = Depends(get_user_id)):
    """
    Streaming via Server-Sent Events.
    O frontend recebe as palavras em tempo real conforme o LLM gera.
    Formato SSE: cada linha começa com "data: " seguido de JSON.
    """
    db_save_message(body.session_id, "user", body.message)

    msgs = db_load_messages(body.session_id)
    new_title = None
    if len(msgs) == 1:
        new_title = auto_title(body.message)
        db_rename_session(body.session_id, new_title)

    filled, sources, graph_ctx = build_prompt_and_retrieve(
        body.message, user_id, body.session_id, body
    )

    full_answer = []

    async def generate():
        # 1. Envia metadados primeiro (fontes, grafo, título novo)
        meta = json.dumps({
            "type": "meta",
            "sources": sources,
            "graph_context": graph_ctx,
            "new_title": new_title,
        }, ensure_ascii=False)
        yield f"data: {meta}\n\n"

        # 2. Streaming do LLM palavra por palavra
        if filled:
            for chunk in llm.stream(filled):
                token = chunk.content
                if token:
                    full_answer.append(token)
                    payload = json.dumps({"type": "token", "token": token}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0)  # permite o event loop respirar
        else:
            mem_key = f"{user_id}_{body.session_id}"
            if mem_key not in memory_cache:
                memory_cache[mem_key] = []
            memory_cache[mem_key].append(HumanMessage(content=body.message))
            for chunk in llm.stream(memory_cache[mem_key]):
                token = chunk.content
                if token:
                    full_answer.append(token)
                    payload = json.dumps({"type": "token", "token": token}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0)
            memory_cache[mem_key].append(AIMessage(content="".join(full_answer)))

        # 3. Sinal de fim
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # 4. Salva resposta completa no banco
        db_save_message(body.session_id, "assistant", "".join(full_answer))

    return StreamingResponse(generate(), media_type="text/event-stream")

# ── Exportar conversa ─────────────────────────────────────────────
@app.get("/sessions/{session_id}/export")
async def export_session(session_id: str, fmt: str = "txt", user_id: str = Depends(get_user_id)):
    """
    Exporta o histórico da conversa como TXT ou Markdown.
    fmt = "txt" ou "md"
    """
    msgs = db_load_messages(session_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Conversa vazia")

    # Busca nome da sessão
    try:
        r = supabase.table("chat_sessions").select("name").eq("id", session_id).limit(1).execute()
        session_name = r.data[0]["name"] if r.data else "Conversa"
    except Exception:
        session_name = "Conversa"

    if fmt == "md":
        lines = [f"# {session_name}\n"]
        for m in msgs:
            role = "**Você**" if m["role"] == "user" else "**Assistente**"
            lines.append(f"{role}\n{m['content']}\n")
        content = "\n---\n".join(lines)
        filename = f"{session_name}.md"
    else:
        lines = [f"{session_name}\n{'='*40}\n"]
        for m in msgs:
            role = "Você" if m["role"] == "user" else "Assistente"
            lines.append(f"[{role}]\n{m['content']}\n")
        content = "\n".join(lines)
        filename = f"{session_name}.txt"

    return PlainTextResponse(
        content=content,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# ── Uploads (com UUID para segurança) ────────────────────────────
@app.post("/upload")
@limiter.limit("20/minute")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_user_id)
):
    folder = user_storage_dir(user_id) / "uploads"
    index = load_uploads_index(user_id)
    saved = []

    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            continue

        content = await f.read()

        # Limitar tamanho: 20MB por arquivo
        if len(content) > 20 * 1024 * 1024:
            continue

        # Nome seguro com UUID — previne path traversal
        safe_name = f"{uuid.uuid4()}{suffix}"
        dest = folder / safe_name
        with open(dest, "wb") as g:
            g.write(content)

        # Mapeia nome_original → nome_seguro
        index[f.filename] = safe_name
        saved.append(f.filename)

        try:
            key = f"{user_id}/{int(time.time())}_{safe_name}"
            supabase.storage.from_("user_uploads").upload(
                key, content, file_options={"content-type": f.content_type}
            )
        except Exception:
            pass

    save_uploads_index(user_id, index)
    return {"saved": saved}

@app.get("/uploads")
async def list_uploads(user_id: str = Depends(get_user_id)):
    index = load_uploads_index(user_id)
    return {"files": list(index.keys())}

@app.delete("/uploads/{filename}")
async def delete_upload(filename: str, user_id: str = Depends(get_user_id)):
    index = load_uploads_index(user_id)
    safe_name = index.pop(filename, None)
    if safe_name:
        dest = user_storage_dir(user_id) / "uploads" / safe_name
        if dest.exists():
            dest.unlink()
        save_uploads_index(user_id, index)
    return {"ok": True}

@app.post("/rebuild-index")
async def rebuild_index(session_id: str, user_id: str = Depends(get_user_id)):
    build_or_load_vectors(user_id, session_id, force=True)
    build_graph(user_id, session_id, force=True)
    return {"ok": True}

# ── Perfil ────────────────────────────────────────────────────────
@app.get("/profile")
async def get_profile(user_id: str = Depends(get_user_id)):
    try:
        r = supabase.table("users_profile").select("*").eq("id", user_id).limit(1).execute()
        return r.data[0] if r.data else {}
    except Exception:
        return {}

@app.patch("/profile")
async def update_profile(body: ProfileUpdate, user_id: str = Depends(get_user_id)):
    payload = {"id": user_id}
    if body.full_name is not None:
        payload["full_name"] = body.full_name
    supabase.table("users_profile").upsert(payload).execute()
    return {"ok": True}

@app.post("/profile/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id)
):
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Avatar muito grande (máx 5MB)")
    filename = f"{user_id}_{uuid.uuid4()}{Path(file.filename).suffix}"
    try:
        supabase.storage.from_("avatars").upload(
            filename, content, file_options={"content-type": file.content_type}
        )
        url_obj = supabase.storage.from_("avatars").get_public_url(filename)
        url = url_obj if isinstance(url_obj, str) else (url_obj.get("publicUrl") or "")
        supabase.table("users_profile").upsert({"id": user_id, "avatar_url": url}).execute()
        return {"avatar_url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))