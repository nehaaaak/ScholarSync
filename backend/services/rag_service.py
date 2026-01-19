import google.generativeai as genai
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-2.5-flash"):
        """Initialize RAG service"""
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 200
            }
        )
    
    def answer_from_paper(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        paper_title: str,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer based on retrieved chunks from a single paper
        
        Args:
            question: User's question
            chunks: Retrieved chunks from vector search (max 6)
            paper_title: Title of the paper
            paper_metadata: Optional metadata (authors, year, etc.)
            
        Returns:
            Dict with answer, confidence, citations
        """
        try:
            # Build context from chunks
            context = self._build_context_from_chunks(chunks)
            
            # Build prompt
            prompt = self._build_per_paper_prompt(
                question=question,
                context=context,
                paper_title=paper_title,
                paper_metadata=paper_metadata
            )
            
            # Generate answer
            response = self.model.generate_content(prompt)
            answer_text = response.text
            
            # Extract citations (which sections were most relevant)
            citations = self._extract_citations(chunks)
            
            # Calculate confidence based on chunk relevance scores
            confidence = self._calculate_confidence(chunks)
            
            return {
                "answer": answer_text,
                "confidence": confidence,
                "citations": citations,
                "chunks_used": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error generating the answer: {str(e)}",
                "confidence": 0.0,
                "citations": [],
                "chunks_used": 0,
                "error": str(e)
            }
    
    def answer_from_multiple_papers(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_papers: int = 5
    ) -> Dict[str, Any]:
        """
        Generate synthesized answer from multiple papers
        
        Args:
            question: User's question
            chunks: Retrieved chunks from multiple papers (max 15)
            max_papers: Maximum number of papers to include
            
        Returns:
            Dict with answer, papers cited, insights
        """
        try:
            # Group chunks by paper
            papers_dict = self._group_chunks_by_paper(chunks)
            
            # Limit to top N papers
            papers_dict = dict(list(papers_dict.items())[:max_papers])
            
            # Build context
            context = self._build_context_from_multiple_papers(papers_dict)
            
            # Build prompt
            prompt = self._build_global_prompt(
                question=question,
                context=context,
                papers_dict=papers_dict
            )
            
            # Generate answer
            response = self.model.generate_content(prompt)
            answer_text = response.text
            
            # Extract insights
            papers_cited = list(papers_dict.keys())
            insights = self._extract_insights(papers_dict)
            
            return {
                "answer": answer_text,
                "papers_cited": papers_cited,
                "total_papers": len(papers_cited),
                "insights": insights,
                "chunks_used": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error: {str(e)}",
                "papers_cited": [],
                "total_papers": 0,
                "insights": [],
                "error": str(e)
            }
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from chunks"""
        context_parts = []
        
        for idx, chunk in enumerate(chunks, 1):
            section = chunk.get('section', 'unknown')
            text = chunk.get('text', '')[:400]
            page = chunk.get('page', '?')
            
            context_parts.append(f"[Section {idx}: {section} - Page {page}]")
            context_parts.append(text)
            context_parts.append("")  # Empty line between sections
        
        return "\n".join(context_parts)
    
    def _build_context_from_multiple_papers(
        self,
        papers_dict: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Build context from multiple papers"""
        context_parts = []
        
        for paper_title, paper_chunks in papers_dict.items():
            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"PAPER: {paper_title}")
            context_parts.append(f"{'='*60}\n")
            
            for idx, chunk in enumerate(paper_chunks, 1):
                section = chunk.get('section', 'unknown')
                text = chunk.get('text', '')[:500]
                
                context_parts.append(f"[Excerpt {idx} - {section}]")
                context_parts.append(text)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_per_paper_prompt(
        self,
        question: str,
        context: str,
        paper_title: str,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for per-paper Q&A"""
        
        metadata_str = ""
        if paper_metadata:
            authors = paper_metadata.get('authors', [])
            year = paper_metadata.get('year', 'Unknown')
            metadata_str = f"\nAuthors: {', '.join(authors[:3]) if authors else 'Unknown'}\nYear: {year}\n"
        
        # prompt = f"""You are a helpful research assistant analyzing a specific research paper.

        #         Paper Title: {paper_title}{metadata_str}

        #         I will provide you with excerpts from this paper, and you will answer questions based ONLY on the information in these excerpts.

        #         EXCERPTS FROM PAPER:
        #         {context}

        #         USER QUESTION: {question}

        #         INSTRUCTIONS:
        #         1. Answer the question using ONLY information from the provided excerpts
        #         2. If the answer is not in the excerpts, clearly state "This information is not mentioned in the paper"
        #         3. Cite which section(s) you're using (e.g., "According to the abstract..." or "In the methodology section...")
        #         4. Be concise and direct - provide the key information without unnecessary elaboration
        #         5. If there are multiple relevant points, organize them with bullet points
        #         6. Do not make assumptions or add information not present in the excerpts

        #         ANSWER:"""

        prompt = f"""You are a research assistant answering questions strictly from provided excerpts.

                Paper: {paper_title}{metadata_str}

                Excerpts:
                {context}

                Question: {question}

                Rules:
                - Use only the excerpts
                - If not found, say so
                - Be concise (2–4 sentences)
                - Cite sections if relevant

                Answer:"""
        
        return prompt
    
    def _build_global_prompt(
        self,
        question: str,
        context: str,
        papers_dict: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Build prompt for cross-paper Q&A"""
        
        papers_list = "\n".join([f"• {title}" for title in papers_dict.keys()])
        
        prompt = f"""You are a research assistant analyzing multiple research papers to answer a question.

                PAPERS BEING ANALYZED:
                {papers_list}

                RELEVANT EXCERPTS FROM PAPERS:
                {context}

                USER QUESTION: {question}

                INSTRUCTIONS:
                1. Synthesize information from the multiple papers provided
                2. Compare and contrast approaches/findings when relevant
                3. Identify common themes, datasets, or methodologies
                4. Cite which paper you're referring to (e.g., "According to [Paper Title]...")
                5. If papers disagree or have different approaches, mention this
                6. Organize your answer clearly with:
                - Main answer/summary
                - Key findings from different papers
                - Notable differences or commonalities
                7. Be comprehensive but concise - focus on the most important information
                8. If the question cannot be fully answered from these papers, acknowledge this

                SYNTHESIZED ANSWER:"""
        
        return prompt
    
    def _group_chunks_by_paper(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by paper title"""
        papers_dict = {}
        
        for chunk in chunks:
            paper_title = chunk.get('paper_title', 'Unknown Paper')
            if paper_title not in papers_dict:
                papers_dict[paper_title] = []
            papers_dict[paper_title].append(chunk)
        
        return papers_dict
    
    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citation information from chunks"""
        citations = []
        
        for chunk in chunks[:3]:  # Top 3 most relevant
            citations.append({
                "section": chunk.get('section', 'unknown'),
                "page": chunk.get('page', '?'),
                "relevance": chunk.get('score', 0.0),
                "text_preview": chunk.get('text', '')[:150] + "..."
            })
        
        return citations
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on chunk relevance
        
        High confidence: Top chunks have high similarity scores
        Low confidence: Chunks have low similarity or few relevant chunks
        """
        if not chunks:
            return 0.0
        
        # Get scores from top 3 chunks
        scores = [chunk.get('score', 0.0) for chunk in chunks[:3]]
        
        if not scores:
            return 0.0
        
        # Average of top scores
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range (similarity scores are typically 0-1 already)
        confidence = min(avg_score, 1.0)
        
        return round(confidence, 2)
    
    def _extract_insights(
        self,
        papers_dict: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract key insights about the papers"""
        insights = []
        
        # Number of papers analyzed
        insights.append(f"Analyzed {len(papers_dict)} papers")
        
        # Most cited paper (most chunks)
        if papers_dict:
            most_cited = max(papers_dict.items(), key=lambda x: len(x[1]))
            insights.append(f"Most relevant paper: {most_cited[0]} ({len(most_cited[1])} excerpts)")
        
        return insights
    
    