"""
eval_rag.py — Avaliação completa das técnicas RAG
Mede: qualidade (RAGAS), velocidade, custo assintótico e custo-benefício

USO:
    python eval_rag.py

SAÍDA:
    - Tabela comparativa no terminal
    - eval_results.json  (dados brutos)
    - eval_report.html   (relatório visual)
"""

import time
import json
import asyncio
import statistics
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# ── LangChain / NVIDIA ────────────────────────────────────────────
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import networkx as nx
import pickle

try:
    import spacy
    nlp_spacy = spacy.load("pt_core_news_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy não disponível — Knowledge Graph desabilitado")

try:
    from sentence_transformers import CrossEncoder
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False
    print("⚠️  CrossEncoder não disponível — Reranking desabilitado")

# RAGAS removido — métricas implementadas diretamente com o LLM NVIDIA
RAGAS_AVAILABLE = False  # não usado mais, mantido por compatibilidade

# ─────────────────────────────────────────────────────────────────
# DATASET DE AVALIAÇÃO
# Perguntas + respostas esperadas (ground truth)
# Edite para usar perguntas relevantes aos seus documentos
# ─────────────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    {
        "question": "Quando foi criado o INPE e qual é sua missão?",
        "ground_truth": "O INPE foi criado em 1961 como GOCNAE e transformado em INPE em 1971. Sua missão é produzir ciência e tecnologia nas áreas espacial e do ambiente terrestre, oferecendo produtos e serviços em benefício do Brasil."
    },
    {
        "question": "O que é o programa CBERS e quais países participam?",
        "ground_truth": "O CBERS, China-Brazil Earth Resources Satellite, é um programa de satélites de sensoriamento remoto desenvolvido em parceria entre Brasil e China desde 1988. Os satélites fornecem imagens para monitoramento ambiental, mapeamento territorial e gestão de recursos naturais."
    },
    {
        "question": "O que é o sistema PRODES e como ele funciona?",
        "ground_truth": "O PRODES é o Projeto de Monitoramento do Desmatamento na Amazônia Legal por Satélite, operando desde 1988. Utiliza imagens de satélite para calcular anualmente a taxa de desmatamento na Amazônia, considerando apenas a remoção completa da cobertura florestal original."
    },
    {
        "question": "Qual a diferença entre o PRODES e o DETER?",
        "ground_truth": "O PRODES realiza análises anuais com alta precisão enquanto o DETER emite alertas rápidos de desmatamento em intervalos de dias. O PRODES é referência oficial de desmatamento e o DETER permite ações de fiscalização mais ágeis pelos órgãos ambientais."
    },
    {
        "question": "Onde fica a sede do INPE e quais outras unidades ele possui?",
        "ground_truth": "A sede principal do INPE está em São José dos Campos em São Paulo. O instituto possui unidades em Natal, Belém, Cachoeira Paulista, Cuiabá, Eusébio e Porto Alegre."
    },
    {
        "question": "O que é o CPTEC e onde está localizado?",
        "ground_truth": "O Centro de Previsão de Tempo e Estudos Climáticos está localizado em Cachoeira Paulista no interior de São Paulo. Opera um dos maiores supercomputadores da América Latina dedicados à modelagem numérica da atmosfera e dos oceanos, gerando previsões de curto prazo quatro vezes ao dia."
    },
    {
        "question": "Quais satélites brasileiros foram desenvolvidos inteiramente no Brasil?",
        "ground_truth": "Os satélites SCD-1 e SCD-2 foram desenvolvidos inteiramente no Brasil e são responsáveis pela coleta de dados ambientais de plataformas distribuídas pelo território nacional, coletando dados meteorológicos, hidrológicos e de qualidade do ar."
    },
    {
        "question": "Como o INPE contribui para a formação científica no Brasil?",
        "ground_truth": "O INPE mantém programa de pós-graduação com cursos de mestrado e doutorado avaliados com notas máximas pela CAPES nas áreas de astrofísica, meteorologia, sensoriamento remoto e outras. Também oferece programas de iniciação científica para estudantes de graduação financiados pelo CNPq e FAPESP."
    },
]

# ─────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES DAS TÉCNICAS A COMPARAR
# ─────────────────────────────────────────────────────────────────
TECHNIQUE_CONFIGS = {
    "Baseline (só FAISS)": {
        "use_hyde": False,
        "use_multi_query": False,
        "use_reranking": False,
        "use_graph": False,
    },
    "HyDE": {
        "use_hyde": True,
        "use_multi_query": False,
        "use_reranking": False,
        "use_graph": False,
    },
    "Multi-Query": {
        "use_hyde": False,
        "use_multi_query": True,
        "use_reranking": False,
        "use_graph": False,
    },
    "Reranking": {
        "use_hyde": False,
        "use_multi_query": False,
        "use_reranking": True,
        "use_graph": False,
    },
    "Knowledge Graph": {
        "use_hyde": False,
        "use_multi_query": False,
        "use_reranking": False,
        "use_graph": True,
    },
    "HyDE + Reranking": {
        "use_hyde": True,
        "use_multi_query": False,
        "use_reranking": True,
        "use_graph": False,
    },
    "Multi-Query + Reranking": {
        "use_hyde": False,
        "use_multi_query": True,
        "use_reranking": True,
        "use_graph": False,
    },
    "Todas as técnicas": {
        "use_hyde": True,
        "use_multi_query": True,
        "use_reranking": True,
        "use_graph": True,
    },
}

# ─────────────────────────────────────────────────────────────────
# COMPLEXIDADE ASSINTÓTICA TEÓRICA
# ─────────────────────────────────────────────────────────────────
COMPLEXITY = {
    "Baseline (só FAISS)":      {"retrieval": "O(d×k)",     "extra_llm_calls": 0, "local_compute": "O(1)"},
    "HyDE":                     {"retrieval": "O(d×k)",     "extra_llm_calls": 1, "local_compute": "O(1)"},
    "Multi-Query":              {"retrieval": "O(N×d×k)",   "extra_llm_calls": 1, "local_compute": "O(N×k)"},
    "Reranking":                {"retrieval": "O(d×k)",     "extra_llm_calls": 0, "local_compute": "O(k×L)"},
    "Knowledge Graph":          {"retrieval": "O(d×k)",     "extra_llm_calls": 0, "local_compute": "O(V+E)"},
    "HyDE + Reranking":         {"retrieval": "O(d×k)",     "extra_llm_calls": 1, "local_compute": "O(k×L)"},
    "Multi-Query + Reranking":  {"retrieval": "O(N×d×k)",   "extra_llm_calls": 1, "local_compute": "O(N×k+k×L)"},
    "Todas as técnicas":        {"retrieval": "O(N×d×k)",   "extra_llm_calls": 2, "local_compute": "O(N×k+k×L+V+E)"},
}
# Legenda:
# d = dimensão do embedding (1024 para NVIDIA)
# k = número de candidatos recuperados (8 no config)
# N = número de queries no multi-query (3+1=4)
# L = tamanho médio dos chunks em tokens (400)
# V = vértices no grafo de conhecimento
# E = arestas no grafo de conhecimento

# ─────────────────────────────────────────────────────────────────
# DATACLASSES DE RESULTADO
# ─────────────────────────────────────────────────────────────────
@dataclass
class StepTiming:
    hyde_ms: float = 0.0
    multi_query_ms: float = 0.0
    reranking_ms: float = 0.0
    graph_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

@dataclass
class QuestionResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    timing: StepTiming
    llm_calls: int
    tokens_used: int

@dataclass
class TechniqueResult:
    name: str
    config: Dict
    questions: List[QuestionResult] = field(default_factory=list)
    # Médias calculadas depois
    avg_timing: Optional[StepTiming] = None
    ragas_scores: Dict = field(default_factory=dict)
    cost_benefit: float = 0.0

# ─────────────────────────────────────────────────────────────────
# SETUP DOS MODELOS
# ─────────────────────────────────────────────────────────────────
print("🔧 Inicializando modelos...")
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
emb = NVIDIAEmbeddings()

# ─────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DOCUMENTOS E ÍNDICES
# ─────────────────────────────────────────────────────────────────
def load_and_index_documents(docs_path: str = "docs") -> Optional[FAISS]:
    """Carrega documentos e cria índice FAISS"""
    docs = []
    folder = Path(docs_path)
    if not folder.exists():
        print(f"⚠️  Pasta '{docs_path}' não encontrada")
        return None

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
            for d in loaded:
                d.metadata["filename"] = f.name
            docs.extend(loaded)
            print(f"   ✅ Carregado: {f.name} ({len(loaded)} páginas)")
        except Exception as e:
            print(f"   ❌ Erro em {f.name}: {e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"   📄 Total de chunks: {len(chunks)}")

    vectors = FAISS.from_documents(chunks, emb)
    return vectors


def build_knowledge_graph(docs_path: str = "docs") -> nx.DiGraph:
    """Constrói Knowledge Graph dos documentos"""
    G = nx.DiGraph()
    if not SPACY_AVAILABLE:
        return G

    folder = Path(docs_path)
    if not folder.exists():
        return G

    docs = []
    for f in folder.iterdir():
        if f.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(f))
                docs.extend(loader.load())
            except Exception:
                pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    import re
    for doc in chunks:
        clean = re.sub(r'\*+', '', doc.page_content)
        clean = re.sub(r'#+\s*', '', clean)
        try:
            spacy_doc = nlp_spacy(clean[:50000])
            entities = [(e.text.strip(), e.label_) for e in spacy_doc.ents if len(e.text.strip()) > 1]
        except Exception:
            entities = []

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

    print(f"   🕸️  Knowledge Graph: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")
    return G

# ─────────────────────────────────────────────────────────────────
# FUNÇÕES RAG (mesmas do main.py)
# ─────────────────────────────────────────────────────────────────
def hyde_query(query: str) -> str:
    try:
        return llm.invoke(
            f"Escreva um parágrafo de 3 frases que responderia: {query}\nApenas o parágrafo:"
        ).content.strip()
    except Exception:
        return query

def multi_query_search(query: str, vectors: FAISS) -> List:
    try:
        prompt = f"""Gere 3 variações diferentes desta pergunta para busca em documentos.
Retorne apenas as 3 perguntas, uma por linha, sem numeração:
Pergunta original: {query}"""
        variations_text = llm.invoke(prompt).content.strip()
        queries = [query] + [v.strip() for v in variations_text.split("\n") if v.strip()][:3]

        all_docs, seen = [], set()
        for q in queries:
            for doc in vectors.similarity_search(q, k=4):
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        return all_docs
    except Exception:
        return vectors.similarity_search(query, k=8)

def rerank_docs(query: str, docs: list, top_n: int = 4) -> list:
    if not RERANKER_AVAILABLE or not docs:
        return docs[:top_n]
    try:
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker_model.predict(pairs)
        ranked = sorted(zip(scores, docs), reverse=True)
        return [d for _, d in ranked[:top_n]]
    except Exception:
        return docs[:top_n]

def query_graph_context(G: nx.DiGraph, query: str) -> str:
    if G.number_of_nodes() == 0:
        return ""
    if SPACY_AVAILABLE:
        try:
            spacy_doc = nlp_spacy(query)
            entities = [(e.text.strip(), e.label_) for e in spacy_doc.ents]
        except Exception:
            entities = []
    else:
        entities = []

    if not entities:
        words = set(query.lower().split())
        entities = [(n, "") for n in G.nodes() if any(w in n.lower() for w in words if len(w) > 3)]

    if not entities:
        return ""

    relations, visited = [], set()
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

PROMPT = ChatPromptTemplate.from_template("""
Você é um assistente útil. Use o contexto para responder de forma direta.
Se não encontrar, diga: "Não encontrei essa informação nos documentos."

Contexto:
{context}

{graph_context}

Pergunta: {input}
Resposta:""")

# ─────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL DE PIPELINE COM MEDIÇÃO
# ─────────────────────────────────────────────────────────────────
def run_pipeline(
    question: str,
    ground_truth: str,
    vectors: FAISS,
    graph: nx.DiGraph,
    config: Dict,
) -> QuestionResult:
    """
    Executa o pipeline RAG com as técnicas configuradas
    e mede o tempo de cada etapa individualmente
    """
    timing = StepTiming()
    llm_calls = 0
    total_start = time.perf_counter()

    # 1. HyDE
    if config["use_hyde"]:
        t = time.perf_counter()
        search_q = hyde_query(question)
        timing.hyde_ms = (time.perf_counter() - t) * 1000
        llm_calls += 1
    else:
        search_q = question

    # 2. Multi-Query
    if config["use_multi_query"]:
        t = time.perf_counter()
        candidate_docs = multi_query_search(search_q, vectors)
        timing.multi_query_ms = (time.perf_counter() - t) * 1000
        llm_calls += 1
    else:
        candidate_docs = vectors.similarity_search(search_q, k=8)

    # 3. Reranking
    if config["use_reranking"]:
        t = time.perf_counter()
        final_docs = rerank_docs(question, candidate_docs, top_n=4)
        timing.reranking_ms = (time.perf_counter() - t) * 1000
    else:
        final_docs = candidate_docs[:4]

    # 4. Knowledge Graph
    if config["use_graph"]:
        t = time.perf_counter()
        graph_ctx = query_graph_context(graph, question)
        timing.graph_ms = (time.perf_counter() - t) * 1000
    else:
        graph_ctx = ""

    # 5. LLM final
    ctx_text = "\n\n".join(
        [f"[{d.metadata.get('filename','?')}]\n{d.page_content}" for d in final_docs]
    )
    filled = PROMPT.format_messages(
        context=ctx_text,
        graph_context=graph_ctx,
        input=question,
    )
    t = time.perf_counter()
    answer = llm.invoke(filled).content
    timing.llm_ms = (time.perf_counter() - t) * 1000
    llm_calls += 1

    timing.total_ms = (time.perf_counter() - total_start) * 1000

    # Estimar tokens (aprox: 1 token ≈ 4 chars)
    total_chars = sum(len(d.page_content) for d in final_docs) + len(question) + len(answer)
    tokens_used = total_chars // 4

    return QuestionResult(
        question=question,
        answer=answer,
        contexts=[d.page_content for d in final_docs],
        ground_truth=ground_truth,
        timing=timing,
        llm_calls=llm_calls,
        tokens_used=tokens_used,
    )

# ─────────────────────────────────────────────────────────────────
# AVALIAÇÃO RAGAS
# ─────────────────────────────────────────────────────────────────
def evaluate_with_ragas(results: List[QuestionResult]) -> Dict:
    """
    Avalia qualidade das respostas usando métricas implementadas diretamente
    com o LLM NVIDIA — sem depender do RAGAS que tem incompatibilidades.

    Métricas implementadas:
    - faithfulness:      a resposta é fiel ao contexto recuperado? (LLM julga)
    - answer_relevancy: a resposta é relevante para a pergunta? (overlap semântico)
    - context_recall:   o contexto cobre o ground truth? (LLM julga)
    - context_precision: os chunks recuperados são precisos? (overlap)
    """
    """
    Métricas baseadas em texto — rápidas, sem chamadas extras ao LLM.

    faithfulness:      % de frases da resposta que aparecem no contexto (token overlap)
    answer_relevancy:  % de palavras-chave da pergunta presentes na resposta
    context_recall:    % de palavras do ground truth cobertas pelo contexto
    context_precision: % de palavras do contexto que são relevantes ao ground truth
    """
    STOPWORDS = {
        "o","a","os","as","um","uma","de","do","da","dos","das","em","no","na",
        "nos","nas","que","é","para","com","se","por","ao","aos","à","às","ou",
        "e","mas","não","seu","sua","seus","suas","este","esta","esse","essa",
        "isso","isto","aqui","ali","como","mais","também","já","quando","onde",
        "qual","quais","são","foi","ser","ter","tem","há","pelo","pela","sobre"
    }

    def tokens(text: str) -> set:
        words = set(text.lower().split())
        return words - STOPWORDS

    faithfulness_scores, relevancy_scores, recall_scores, precision_scores = [], [], [], []

    for r in results:
        ctx_text = " ".join(r.contexts)
        ctx_tok      = tokens(ctx_text)
        answer_tok   = tokens(r.answer)
        question_tok = tokens(r.question)
        truth_tok    = tokens(r.ground_truth)

        # Faithfulness: % das palavras da resposta que estão no contexto
        faith = len(answer_tok & ctx_tok) / max(len(answer_tok), 1)
        faithfulness_scores.append(min(1.0, faith))

        # Answer Relevancy: % das palavras-chave da pergunta na resposta
        rel = len(answer_tok & question_tok) / max(len(question_tok), 1)
        relevancy_scores.append(min(1.0, rel * 1.5))

        # Context Recall: % do ground truth coberto pelo contexto
        recall = len(truth_tok & ctx_tok) / max(len(truth_tok), 1)
        recall_scores.append(min(1.0, recall))

        # Context Precision: % do contexto relevante ao ground truth
        prec = len(ctx_tok & truth_tok) / max(len(ctx_tok), 1)
        precision_scores.append(min(1.0, prec * 4.0))

    return {
        "faithfulness":      round(statistics.mean(faithfulness_scores), 4),
        "answer_relevancy":  round(statistics.mean(relevancy_scores), 4),
        "context_recall":    round(statistics.mean(recall_scores), 4),
        "context_precision": round(statistics.mean(precision_scores), 4),
    }

# ─────────────────────────────────────────────────────────────────
# CÁLCULO DO CUSTO-BENEFÍCIO
# ─────────────────────────────────────────────────────────────────
def compute_cost_benefit(ragas_scores: Dict, avg_timing: StepTiming, llm_calls_avg: float) -> Dict:
    """
    Score composto de custo-benefício:

    quality_score  = média das métricas RAGAS disponíveis (0 a 1)
    cost_score     = tempo_total_normalizado + peso × chamadas_llm_extra
    cost_benefit   = quality_score / cost_score

    Quanto MAIOR o cost_benefit, melhor a técnica em relação ao custo.
    """
    # Qualidade: média das métricas disponíveis
    available = [v for v in ragas_scores.values() if v is not None and isinstance(v, float)]
    quality = statistics.mean(available) if available else 0.0

    # Custo: tempo total em segundos + penalidade por chamadas LLM extra
    # Cada chamada extra ao LLM custa ~2s de latência de API
    time_s = avg_timing.total_ms / 1000
    llm_penalty = (llm_calls_avg - 1) * 2.0  # baseline tem 1 chamada
    cost = time_s + llm_penalty

    cost_benefit = round(quality / cost if cost > 0 else 0.0, 4)

    return {
        "quality_score":   round(quality, 4),
        "total_time_s":    round(time_s, 3),
        "llm_calls_avg":   round(llm_calls_avg, 2),
        "cost_score":      round(cost, 3),
        "cost_benefit":    cost_benefit,
    }

# ─────────────────────────────────────────────────────────────────
# RELATÓRIO HTML
# ─────────────────────────────────────────────────────────────────
def generate_html_report(all_results: List[Dict]) -> str:
    rows_quality, rows_timing, rows_complexity, rows_cb = "", "", "", ""

    for r in all_results:
        name = r["name"]
        rs   = r.get("ragas_scores", {})
        at   = r.get("avg_timing", {})
        cb   = r.get("cost_benefit_details", {})
        cplx = COMPLEXITY.get(name, {})

        def f(v): return f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else str(v))
        def ms(v): return f"{v:.0f}ms" if isinstance(v, float) and v > 0 else "—"

        rows_quality += f"""<tr>
            <td><b>{name}</b></td>
            <td>{f(rs.get('faithfulness'))}</td>
            <td>{f(rs.get('answer_relevancy'))}</td>
            <td>{f(rs.get('context_recall'))}</td>
            <td>{f(rs.get('context_precision'))}</td>
        </tr>"""

        rows_timing += f"""<tr>
            <td><b>{name}</b></td>
            <td>{ms(at.get('hyde_ms',0))}</td>
            <td>{ms(at.get('multi_query_ms',0))}</td>
            <td>{ms(at.get('reranking_ms',0))}</td>
            <td>{ms(at.get('graph_ms',0))}</td>
            <td>{ms(at.get('llm_ms',0))}</td>
            <td><b>{ms(at.get('total_ms',0))}</b></td>
        </tr>"""

        rows_complexity += f"""<tr>
            <td><b>{name}</b></td>
            <td><code>{cplx.get('retrieval','—')}</code></td>
            <td><code>{cplx.get('local_compute','—')}</code></td>
            <td>{cplx.get('extra_llm_calls','—')}</td>
        </tr>"""

        cb_score = cb.get('cost_benefit', 0)
        bar = int(cb_score * 300)
        rows_cb += f"""<tr>
            <td><b>{name}</b></td>
            <td>{f(cb.get('quality_score'))}</td>
            <td>{cb.get('total_time_s','—')}s</td>
            <td>{cb.get('llm_calls_avg','—')}</td>
            <td>{cb.get('cost_score','—')}</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px">
                    <div style="width:{min(bar,300)}px;height:16px;background:linear-gradient(90deg,#4f8ef7,#a78bfa);border-radius:4px"></div>
                    <b>{f(cb_score)}</b>
                </div>
            </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Avaliação RAG — NIM Chat</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 32px; }}
  h1 {{ font-size: 28px; background: linear-gradient(90deg,#4f8ef7,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  h2 {{ font-size: 18px; color: #a78bfa; margin-top: 40px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
  th {{ background: #161b22; color: #8b949e; padding: 10px 14px; text-align: left; font-weight: 600; letter-spacing: 0.5px; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #161b22; }}
  code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #79c0ff; }}
  .note {{ font-size: 12px; color: #8b949e; margin-top: 8px; font-style: italic; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
</style>
</head>
<body>
<h1>📊 Avaliação Comparativa das Técnicas RAG</h1>
<p class="note">Gerado automaticamente por eval_rag.py — {len(EVAL_QUESTIONS)} perguntas de avaliação</p>

<h2>🎯 Qualidade das Respostas (RAGAS)</h2>
<p class="note">Escala 0–1. Quanto maior, melhor. Faithfulness = fidelidade ao contexto. Answer Relevancy = relevância para a pergunta.</p>
<table>
  <tr><th>Técnica</th><th>Faithfulness</th><th>Answer Relevancy</th><th>Context Recall</th><th>Context Precision</th></tr>
  {rows_quality}
</table>

<h2>⏱️ Desempenho e Velocidade (média por pergunta)</h2>
<p class="note">Tempo de cada etapa em milissegundos. LLM inclui latência da API NVIDIA.</p>
<table>
  <tr><th>Técnica</th><th>HyDE</th><th>Multi-Query</th><th>Reranking</th><th>Graph</th><th>LLM</th><th>Total</th></tr>
  {rows_timing}
</table>

<h2>🧮 Complexidade Assintótica</h2>
<p class="note">d=dimensão embedding (1024), k=candidatos (8), N=variações multi-query (4), L=tamanho chunk (400 tokens), V=vértices do grafo, E=arestas do grafo</p>
<table>
  <tr><th>Técnica</th><th>Retrieval</th><th>Computação local</th><th>Chamadas LLM extras</th></tr>
  {rows_complexity}
</table>

<h2>⚖️ Custo-Benefício (quality_score / cost_score)</h2>
<p class="note">Quanto MAIOR a barra, melhor a relação qualidade/custo. Cost score = tempo_total_s + (chamadas_llm_extra × 2s)</p>
<table>
  <tr><th>Técnica</th><th>Qualidade</th><th>Tempo total</th><th>Chamadas LLM</th><th>Cost score</th><th>Cost-Benefit ↑</th></tr>
  {rows_cb}
</table>

</body>
</html>"""
    return html

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  📊 AVALIAÇÃO DAS TÉCNICAS RAG — NIM CHAT")
    print("="*60 + "\n")

    # Carrega documentos
    print("📂 Carregando documentos...")
    vectors = load_and_index_documents("docs")
    if vectors is None:
        print("❌ Nenhum documento encontrado em ./docs — abortando")
        return

    print("\n🕸️  Construindo Knowledge Graph...")
    graph = build_knowledge_graph("docs")

    all_results = []

    for tech_name, config in TECHNIQUE_CONFIGS.items():
        print(f"\n{'─'*50}")
        print(f"🔬 Avaliando: {tech_name}")
        print(f"{'─'*50}")

        question_results = []

        for i, qa in enumerate(EVAL_QUESTIONS):
            print(f"   [{i+1}/{len(EVAL_QUESTIONS)}] {qa['question'][:60]}...")
            try:
                result = run_pipeline(
                    question=qa["question"],
                    ground_truth=qa["ground_truth"],
                    vectors=vectors,
                    graph=graph,
                    config=config,
                )
                question_results.append(result)
                print(f"         ✅ {result.timing.total_ms:.0f}ms | {result.llm_calls} chamadas LLM")
            except Exception as e:
                print(f"         ❌ Erro: {e}")

        if not question_results:
            continue

        # Calcula médias de timing
        avg_timing = StepTiming(
            hyde_ms=       statistics.mean(r.timing.hyde_ms for r in question_results),
            multi_query_ms=statistics.mean(r.timing.multi_query_ms for r in question_results),
            reranking_ms=  statistics.mean(r.timing.reranking_ms for r in question_results),
            graph_ms=      statistics.mean(r.timing.graph_ms for r in question_results),
            llm_ms=        statistics.mean(r.timing.llm_ms for r in question_results),
            total_ms=      statistics.mean(r.timing.total_ms for r in question_results),
        )
        avg_llm_calls = statistics.mean(r.llm_calls for r in question_results)

        # RAGAS
        print(f"   📐 Calculando métricas de qualidade...")
        ragas_scores = evaluate_with_ragas(question_results)

        # Custo-benefício
        cb_details = compute_cost_benefit(ragas_scores, avg_timing, avg_llm_calls)

        print(f"   📊 Qualidade: {ragas_scores.get('faithfulness','?')} faithfulness | {ragas_scores.get('answer_relevancy','?')} relevancy")
        print(f"   ⚖️  Cost-Benefit: {cb_details['cost_benefit']}")

        all_results.append({
            "name": tech_name,
            "config": config,
            "avg_timing": asdict(avg_timing),
            "ragas_scores": ragas_scores,
            "cost_benefit_details": cb_details,
            "complexity": COMPLEXITY.get(tech_name, {}),
        })

    # ── Salva JSON ──────────────────────────────────────────────
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultados salvos em eval_results.json")

    # ── Gera HTML ───────────────────────────────────────────────
    html = generate_html_report(all_results)
    with open("eval_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"📄 Relatório HTML gerado em eval_report.html")

    # ── Tabela final no terminal ─────────────────────────────────
    print("\n" + "="*60)
    print("  RANKING POR CUSTO-BENEFÍCIO")
    print("="*60)
    ranking = sorted(all_results, key=lambda x: x["cost_benefit_details"]["cost_benefit"], reverse=True)
    for i, r in enumerate(ranking, 1):
        cb = r["cost_benefit_details"]
        print(f"  {i}. {r['name']:<30} CB={cb['cost_benefit']:.4f}  Q={cb['quality_score']:.4f}  T={cb['total_time_s']:.1f}s")

    print("\n✅ Avaliação concluída!")

if __name__ == "__main__":
    main()