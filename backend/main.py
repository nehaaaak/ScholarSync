from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from qdrant_client import QdrantClient
from datetime import datetime, timezone 
from typing import Dict, Any, List, Optional
import os
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
import math

from services import PDFFetcher, PDFProcessor
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService
from services.langflow_client import LangFlowClient


# Import adapters
from adapters import (
    ArxivAdapter, 
    SemanticScholarAdapter,
    PapersWithCodeAdapter,
    UnpaywallAdapter
)


# Load environment variables
load_dotenv()


# Conversation storage (in-memory for now)
conversation_store = {}

app = FastAPI(
    title="ScholarSync API",
    description="AI-powered research assistant backend",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PAPERS_STORAGE_PATH = os.getenv("PAPERS_STORAGE_PATH", "../storage/papers")

# Initialize adapters and services
arxiv_adapter = ArxivAdapter()
semantic_adapter = SemanticScholarAdapter()
pwc_adapter = PapersWithCodeAdapter()
unpaywall_adapter = UnpaywallAdapter()

pdf_fetcher = PDFFetcher(storage_path=PAPERS_STORAGE_PATH)
pdf_processor = PDFProcessor(
    storage_path=PAPERS_STORAGE_PATH,
    chunks_path=os.getenv("CHUNKS_STORAGE_PATH", "../data/chunks")
)
embedding_service = EmbeddingService(
    gemini_api_key=GEMINI_API_KEY,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    collection_name="papers_collection"
)
rag_service = RAGService(
    gemini_api_key=GEMINI_API_KEY,
    model_name="gemini-2.5-flash"
)
langflow_client = LangFlowClient(
    langflow_url=os.getenv("LANGFLOW_URL", "http://localhost:7860"),
    flow_id=os.getenv("LANGFLOW_FLOW_ID"),  # Set in .env
    api_key=os.getenv("LANGFLOW_API_KEY")  # Optional
)


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    include_code: bool = True
    sort_by: str = "relevance"  # "relevance", "most_recent", "most_cited", "has_code"  #added extra
    offset: int = 0


class DownloadRequest(BaseModel):
    paper_id: str
    paper_title: str
    pdf_url: Optional[str] = None


class EmbedAbstractRequest(BaseModel):
    """Request model for embedding paper abstract only"""
    paper_id: str
    paper_title: str
    abstract: str
    paper_metadata: Dict[str, Any] = {}


class EmbedFullPaperRequest(BaseModel):
    """Request model for embedding full paper"""
    paper_id: str
    paper_title: str
    paper_metadata: Dict[str, Any] = {}


class SearchWithinPaperRequest(BaseModel):
    """Request model for per-paper semantic search"""
    paper_id: str
    query: str
    top_k: int = 6


class GlobalSearchRequest(BaseModel):
    """Request model for global semantic search"""
    query: str
    top_k: int = 15
    filters: Optional[Dict[str, Any]] = None


class AskPaperRequest(BaseModel):
    """Request for per-paper Q&A"""
    paper_id: str
    question: str
    paper_title: str
    paper_metadata: Optional[Dict[str, Any]] = {}


class AskGlobalRequest(BaseModel):
    """Request for global Q&A across papers"""
    question: str
    filters: Optional[Dict[str, Any]] = None
    max_papers: int = 5


class ConversationMessage(BaseModel):
    paper_id: Optional[str] = None
    role: str
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = {}



class SummarizeRequest(BaseModel):
    paper_id: str
    paper_title: Optional[str] = None



def check_gemini_connection() -> Dict[str, Any]:
    """Test Gemini API connection"""
    if not GEMINI_API_KEY:
        return {"status": "not_configured", "message": "API key not set"}
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        models = genai.list_models()
        return {
            "status": "connected",
            "message": "Gemini API operational",
            "models_available": len(list(models))
        }
    except ImportError:
        return {"status": "error", "message": "google-generativeai not installed"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}


def check_qdrant_connection() -> Dict[str, Any]:
    """Test Qdrant connection"""
    if not QDRANT_URL or not QDRANT_API_KEY:
        return {"status": "not_configured", "message": "URL or API key not set"}
    
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections()
        return {
            "status": "connected",
            "message": "Qdrant cluster operational",
            "collections_count": len(collections.collections)
        }
    except ImportError:
        return {"status": "error", "message": "qdrant-client not installed"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}


def merge_and_deduplicate(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge papers from multiple sources and remove duplicates"""
    seen_titles = set()
    seen_dois = set()
    unique_papers = []
    
    for paper in papers:
        title_normalized = paper['title'].lower().strip()
        doi = paper.get('doi')
        
        is_duplicate = False
        
        if doi and doi in seen_dois:
            is_duplicate = True
        
        if title_normalized in seen_titles:
            is_duplicate = True
        
        if not is_duplicate:
            unique_papers.append(paper)
            seen_titles.add(title_normalized)
            if doi:
                seen_dois.add(doi)
    
    return unique_papers


def rank_papers(
    papers: List[Dict[str, Any]],
    sort_by: str = "relevance"
) -> List[Dict[str, Any]]:
    current_year = datetime.now().year

    sort_by = (sort_by or "relevance").lower()

    # Pre-compute scores used by multiple modes
    for paper in papers:
        try:
            year = paper.get("year") or 2000
            citations = paper.get("citation_count") or 0
            has_code = paper.get("has_code", False)
            source = paper.get("source", "")

            # Year score in [0, 1]
            year_score = max(0.0, (year - 2000) / (current_year - 2000))

            # Citation score in [0, 1] using log scale
            citation_score = math.log10(citations + 1) / 4.0
            citation_score = min(citation_score, 1.0)

            # Code bonus (0 or 0.1)
            code_bonus = 0.10 if has_code else 0.0

            # Source bonus (Option B)
            if source == "semantic_scholar":
                source_bonus = 0.15
            elif source == "paperswithcode":
                source_bonus = 0.10
            else:
                source_bonus = 0.0

            # Default relevance score (Option A + B)
            # 0.40 * year + 0.50 * citations + 0.10 * code + source_bonus
            relevance_score = (
                0.40 * year_score +
                0.50 * citation_score +
                code_bonus +         # already scaled to 0.10
                source_bonus
            )

            paper["year_score"] = year_score
            paper["citation_score"] = citation_score
            paper["relevance_score"] = relevance_score
            paper["code_bonus"] = code_bonus
            paper["source_bonus"] = source_bonus

        except Exception as e:
            print(f"Error ranking paper: {e}")
            paper["year_score"] = 0.0
            paper["citation_score"] = 0.0
            paper["relevance_score"] = 0.0
            paper["code_bonus"] = 0.0
            paper["source_bonus"] = 0.0

    # Apply sorting strategy (Option C)
    if sort_by == "most_recent":
        papers.sort(key=lambda x: x.get("year", 0), reverse=True)
    elif sort_by == "most_cited":
        papers.sort(key=lambda x: x.get("citation_count", 0) or 0, reverse=True)
    elif sort_by == "has_code":
        # (has_code True first) then citations desc
        papers.sort(
            key=lambda x: (
                x.get("has_code", False),
                x.get("citation_count", 0) or 0,
            ),
            reverse=True,
        )
    else:
        # "relevance" (recommended default)
        papers.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    return papers


def enrich_with_free_pdfs(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add free PDF links via Unpaywall"""
    for paper in papers:
        if paper.get('doi') and not paper.get('pdf_url'):
            unpaywall_adapter.enrich_paper(paper)
    return papers


@app.get("/")
async def root():
    return {
        "app": "ScholarSync",
        "version": "0.1.0",
        "status": "running",
        "features": [
            "Multi-source search",
            "PDF download & storage",
            "Text extraction & chunking",
            "Vector embeddings (Qdrant)",
            "Per-paper Q&A (RAG)",  
            "Global Q&A (RAG)"
        ]
    }


@app.get("/status")
async def status():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    gemini_status = check_gemini_connection()
    qdrant_status = check_qdrant_connection()
    
    all_healthy = (
        gemini_status["status"] == "connected" and
        qdrant_status["status"] == "connected"
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "backend": {"status": "operational", "message": "FastAPI running"},
            "gemini": gemini_status,
            "qdrant": qdrant_status
        }
    }


@app.post("/search")
async def search_papers(request: SearchRequest):
    """Search for papers across multiple sources"""
    try:
        query = request.query
        max_results = min(request.max_results, 20)
        offset = request.offset
        
        print(f"Searching for: {query} (offset: {offset})")

        # Fetch MORE results than needed (for pagination)
        fetch_count = max_results + offset + 10  # Extra buffer
        
        # Search all sources
        arxiv_results = arxiv_adapter.search(query, max_results=fetch_count)
        semantic_results = semantic_adapter.search(query, max_results=fetch_count)
        
        pwc_results = []
        if request.include_code:
            pwc_results = pwc_adapter.search(query, max_results=fetch_count)
        
        print(f"arXiv: {len(arxiv_results)} results")
        print(f"Semantic Scholar: {len(semantic_results)} results")
        print(f"PapersWithCode: {len(pwc_results)} results")

        sort_by = request.sort_by  

        # Merge all results
        all_papers = arxiv_results + semantic_results + pwc_results
        
        # Deduplicate
        unique_papers = merge_and_deduplicate(all_papers)
        
        # Enrich with free PDFs
        unique_papers = enrich_with_free_pdfs(unique_papers)
        
        # Check which papers are already downloaded
        for paper in unique_papers:
            local_path = pdf_fetcher.get_local_path(paper['id'])
            paper['downloaded'] = local_path is not None
            if local_path:
                paper['local_path'] = local_path
        
        # Rank / sort using requested strategy
        ranked_papers = rank_papers(unique_papers, sort_by=sort_by)

        # Apply pagination
        start_idx = offset
        end_idx = offset + max_results
        # Return top N
        final_results = ranked_papers[start_idx:end_idx]

        # Calculate actual sources in final results (after deduplication)
        actual_sources = {
            'arxiv': sum(1 for p in final_results if p['source'] == 'arxiv'),
            'semantic_scholar': sum(1 for p in final_results if p['source'] == 'semantic_scholar'),
            'paperswithcode': sum(1 for p in final_results if p['source'] == 'paperswithcode')
        }

        return {
            "query": query,
            "total_found": len(final_results),
            "total_available": len(ranked_papers),  
            "offset": offset,
            "has_more": end_idx < len(ranked_papers),  
            "sources": actual_sources,  
            "papers": final_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/get-paper")
async def get_paper(request: DownloadRequest):
    """
    Download and store a paper PDF
    Now auto-processes AND auto-embeds full paper!
    """
    try:
        # Check if already downloaded
        existing_path = pdf_fetcher.get_local_path(request.paper_id)
        if existing_path:
            # Already downloaded, ensure processed and embedded
            if not pdf_processor.is_processed(request.paper_id):
                pdf_processor.process_paper(request.paper_id, request.paper_title)
            
            # Auto-embed full paper (for global Q&A)
            if not embedding_service.is_paper_embedded(request.paper_id, check_full=True):
                chunks = pdf_processor.get_chunks(request.paper_id)
                if chunks:
                    embedding_service.embed_paper_full(
                        paper_id=request.paper_id,
                        paper_title=request.paper_title,
                        chunks=chunks['chunks'],
                        paper_metadata={}
                    )
            
            return {
                "success": True,
                "cached": True,
                "local_path": existing_path,
                "message": "Paper already downloaded"
            }
        
        # Prepare paper dict for fetcher
        paper = {
            'id': request.paper_id,
            'title': request.paper_title,
            'pdf_url': request.pdf_url
        }
        
        # Download PDF
        result = pdf_fetcher.download_pdf(paper)
        
        if result['success']:
            # Auto-process after successful download
            try:
                pdf_processor.process_paper(request.paper_id, request.paper_title)
                
                # Auto-embed full paper (for global Q&A)
                chunks = pdf_processor.get_chunks(request.paper_id)
                if chunks:
                    embedding_service.embed_paper_full(
                        paper_id=request.paper_id,
                        paper_title=request.paper_title,
                        chunks=chunks['chunks'],
                        paper_metadata={}
                    )
                    
            except Exception as e:
                # Don't fail download if processing/embedding fails
                print(f"Auto-processing/embedding failed: {e}")
            
            return {
                "success": True,
                "cached": result.get('cached', False),
                "local_path": result['local_path'],
                "file_size": result.get('file_size', 0),
                "message": "PDF downloaded successfully"
            }
        else:
            return {
                "success": False,
                "paywalled": result.get('paywalled', False),
                "error": result.get('error', 'Unknown error'),
                "message": "Failed to download PDF"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/storage-stats")
async def storage_stats():
    """Get statistics about stored PDFs"""
    try:
        stats = pdf_fetcher.get_storage_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/test-apis")
async def test_apis():
    """Test all external APIs"""
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {}
    }
    
    gemini_result = check_gemini_connection()
    results["tests"]["gemini"] = gemini_result
    
    qdrant_result = check_qdrant_connection()
    results["tests"]["qdrant"] = qdrant_result
    
    all_connected = all(
        test["status"] == "connected" 
        for test in results["tests"].values()
    )
    
    results["overall_status"] = "all_systems_go" if all_connected else "issues_detected"
    results["ready_for_phase_b"] = all_connected
    
    return results


@app.get("/chunks/{paper_id}")
async def get_paper_chunks(paper_id: str):
    try:
        chunks = pdf_processor.get_chunks(paper_id)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="Paper not processed yet"
            )
        
        return chunks
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")


@app.post("/embed-abstract")
async def embed_abstract(request: EmbedAbstractRequest):
    """
    Embed ONLY the abstract of a paper
    Auto-called when user clicks "Ask About This Paper"
    """
    try:
        result = embedding_service.embed_paper_abstract(
            paper_id=request.paper_id,
            paper_title=request.paper_title,
            abstract=request.abstract,
            paper_metadata=request.paper_metadata
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed abstract: {str(e)}")


@app.post("/embed-full-paper")
async def embed_full_paper(request: EmbedFullPaperRequest):
    try:
        #already fully embedded?
        if embedding_service.is_paper_embedded(request.paper_id, check_full=True):
            # Nothing to do – embeddings already in Qdrant
            return {
                "success": True,
                "cached": True,
                "chunks_embedded": 0,
                "message": "Paper already fully embedded; skipped re-embedding."
            }

        #Only now do we touch the PDF / chunking
        chunks = pdf_processor.get_chunks(request.paper_id)

        if not chunks or not chunks.get("chunks"):
            raise HTTPException(
                status_code=404,
                detail=f"Paper {request.paper_id} not processed yet. Please download first."
            )

        #Embed all chunks
        result = embedding_service.embed_paper_full(
            paper_id=request.paper_id,
            paper_title=request.paper_title,
            chunks=chunks["chunks"],
            paper_metadata=request.paper_metadata,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed paper: {str(e)}")


@app.post("/search-within-paper")
async def search_within_paper(request: SearchWithinPaperRequest):
    """
    Semantic search within a specific paper
    Used by per-paper chat interface
    """
    try:
        results = embedding_service.search_within_paper(
            paper_id=request.paper_id,
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "paper_id": request.paper_id,
            "query": request.query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search-global")
async def search_global(request: GlobalSearchRequest):
    """
    Semantic search across ALL downloaded papers
    Used by global research assistant
    """
    try:
        results = embedding_service.search_global(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Group results by paper
        papers_cited = list(set([r['paper_id'] for r in results]))
        
        return {
            "query": request.query,
            "results": results,
            "total_chunks": len(results),
            "papers_cited": papers_cited,
            "total_papers": len(papers_cited)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global search failed: {str(e)}")


@app.get("/embedding-status/{paper_id}")
async def get_embedding_status(paper_id: str):
    """
    Check if a paper has been embedded (abstract and/or full)
    """
    try:
        has_abstract = embedding_service.is_paper_embedded(paper_id, check_full=False)
        has_full = embedding_service.is_paper_embedded(paper_id, check_full=True)
        
        return {
            "paper_id": paper_id,
            "has_abstract": has_abstract,
            "has_full": has_full,
            "ready_for_per_paper_qa": has_abstract,
            "ready_for_global_qa": has_full
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")


@app.delete("/embeddings/{paper_id}")
async def delete_paper_embeddings(paper_id: str):
    """
    Delete all embeddings for a paper
    Useful for cleanup or re-embedding
    """
    try:
        success = embedding_service.delete_paper(paper_id)
        
        if success:
            return {
                "success": True,
                "paper_id": paper_id,
                "message": "Embeddings deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete embeddings")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/collection-stats")
async def get_collection_stats():
    """
    Get statistics about the vector database
    """
    try:
        stats = embedding_service.get_collection_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/ask-paper")
async def ask_paper(request: AskPaperRequest):
    """
    Per-Paper Q&A - Answer questions about a specific paper.
    Uses abstract-only embeddings + RAG.
    """
    try:
        #Ensure abstract is embedded
        has_abstract = embedding_service.is_paper_embedded(
            paper_id=request.paper_id,
            check_full=False  # just abstract
        )

        if not has_abstract:
            raise HTTPException(
                status_code=400,
                detail="Paper abstract not embedded yet. Please embed it first."
            )

        #Semantic search within this paper
        chunks = embedding_service.search_within_paper(
            paper_id=request.paper_id,
            query=request.question,
            top_k=6,
        )

        if not chunks:
            return {
                "answer": (
                    "I couldn't find relevant information in this paper to answer your question."
                ),
                "confidence": 0.0,
                "citations": [],
                "chunks_used": 0,
            }

        #RAG answer from Gemini
        rag_response = rag_service.answer_from_paper(
            question=request.question,
            chunks=chunks,
            paper_title=request.paper_title,
            paper_metadata=request.paper_metadata or {},
        )
        return rag_response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


@app.post("/ask-global")
async def ask_global(request: AskGlobalRequest):
    """
    Global Q&A - Answer questions across all downloaded papers.
    Uses global embeddings + RAG.
    """
    try:
        #Semantic search across all papers
        chunks = embedding_service.search_global(
            query=request.question,
            top_k=15,
            filters=request.filters,
        )

        if not chunks:
            return {
                "answer": (
                    "I couldn't find relevant papers to answer this question. "
                    "Make sure you have downloaded and embedded some papers first."
                ),
                "papers_cited": [],
                "total_papers": 0,
                "insights": [],
            }

        #Synthesized answer across multiple papers
        rag_response = rag_service.answer_from_multiple_papers(
            question=request.question,
            chunks=chunks,
            max_papers=request.max_papers,
        )
        return rag_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global Q&A failed: {str(e)}")


@app.post("/search-within-paper-enhanced")
async def search_within_paper_enhanced(
    paper_id: str = Query(...),
    question: str = Query(...),
    generate_answer: bool = Query(False),
    paper_title: str = Query(None),
):
    """
    Enhanced version - vector search within a paper,
    optionally with a RAG answer bundled.
    """
    try:
        #Semantic search within this paper
        chunks = embedding_service.search_within_paper(
            paper_id=paper_id,
            query=question,
            top_k=6,
        )

        response: Dict[str, Any] = {
            "paper_id": paper_id,
            "query": question,
            "results": chunks,
            "total_found": len(chunks),
        }

        #Optional RAG answer
        if generate_answer and chunks:
            rag_response = rag_service.answer_from_paper(
                question=question,
                chunks=chunks,
                paper_title=paper_title or paper_id,
                paper_metadata={},
            )
            response["answer"] = rag_response

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")


@app.post("/conversation/save")
async def save_conversation_message(message: ConversationMessage):
    """
    Save a conversation message
    """
    try:
        key = message.paper_id if message.paper_id else "global"
        
        if key not in conversation_store:
            conversation_store[key] = []
        
        conversation_store[key].append(message.dict())
        
        return {
            "success": True,
            "message": "Message saved",
            "total_messages": len(conversation_store[key])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")


@app.get("/conversation/history/{paper_id}")
async def get_conversation_history(paper_id: str):
    """
    Get conversation history for a paper
    """
    try:
        history = conversation_store.get(paper_id, [])
        
        return {
            "paper_id": paper_id,
            "messages": history,
            "total_messages": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve: {str(e)}")


@app.delete("/conversation/clear/{paper_id}")
async def clear_conversation(paper_id: str):
    """
    Clear conversation history for a paper
    """
    try:
        if paper_id in conversation_store:
            del conversation_store[paper_id]
        
        return {
            "success": True,
            "message": f"Conversation cleared for {paper_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")


@app.post("/langflow/fetch-paper")
async def fetch_paper_for_langflow(request: dict):
    """
    Endpoint called BY LangFlow to fetch paper data
    """
    paper_id = (
        request.get("paper_id")
        or request.get("text")
        or request.get("input_value")
    )

    if not paper_id:
        raise HTTPException(status_code=400, detail="paper_id missing")
    
    try:
        # Get chunks
        chunks_data = pdf_processor.get_chunks(paper_id)
        
        if not chunks_data:
            raise HTTPException(status_code=404, detail="Paper not found. Please download paper first.")
        
        return {
            "paper_id": paper_id,
            "paper_title": chunks_data.get('paper_title', 'Unknown'),
            "chunks": chunks_data.get('chunks', [])[:8]
        }
        
    except Exception as e:
        print(f" Langflow Fetch Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-to-notion")
async def summarize_to_notion(request: SummarizeRequest):
    """
    Trigger LangFlow → Notion workflow
    """
    try:
        result = langflow_client.trigger_summarization(
            paper_id=request.paper_id,
            paper_title=request.paper_title
        )
        
        # if result['success']:
        if isinstance(result, dict) and result.get('success'):
            return {
                "success": True,
                "paper_id": request.paper_id,
                "notion_url": result.get('notion_url'),
                "message": "✅ Summary saved to Notion!"
            }
        # else:
        #     raise HTTPException(
        #         status_code=500,
        #         detail=result.get('error', 'Summarization failed')
        #     )
        error_detail = result.get('error') if isinstance(result, dict) else "Unknown Langflow error"
        raise HTTPException(status_code=502, detail=f"Langflow Error: {error_detail}")
            
    except Exception as e:
        print(f"CRITICAL ERROR in summarize_to_notion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)