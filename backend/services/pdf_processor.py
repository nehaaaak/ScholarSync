import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

class PDFProcessor:
    """Service for processing PDF papers into chunks"""
    
    def __init__(self, storage_path: str = "./storage/papers", chunks_path: str = "./data/chunks"):
        self.storage_path = Path(storage_path)
        self.chunks_path = Path(chunks_path)
        self.chunks_path.mkdir(parents=True, exist_ok=True)
    
    def process_paper(self, paper_id: str, paper_title: str) -> Dict[str, Any]:
        """
        Process a paper: extract text, identify sections, create chunks
        
        Args:
            paper_id: Unique paper identifier
            paper_title: Paper title
            
        Returns:
            Dict with chunks and metadata
        """
        try:
            # Find PDF file
            pdf_path = self._find_pdf(paper_id)
            if not pdf_path:
                return {
                    "success": False,
                    "error": "PDF file not found"
                }
            
            # Extract text
            full_text, page_texts = self._extract_text(pdf_path)
            
            if not full_text.strip():
                return {
                    "success": False,
                    "error": "Could not extract text from PDF"
                }
            
            # Identify sections
            sections = self._identify_sections(full_text, page_texts)
            
            # Create chunks
            chunks = self._create_chunks(sections, paper_id, paper_title)
            
            # Save chunks to file
            self._save_chunks(paper_id, chunks)
            
            return {
                "success": True,
                "paper_id": paper_id,
                "total_chunks": len(chunks),
                "sections_found": list(sections.keys()),
                "total_text_length": len(full_text),
                "chunks": chunks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }
    
    def _find_pdf(self, paper_id: str) -> Optional[Path]:
        """Find PDF file by paper_id"""
        # Try direct match
        pdf_path = self.storage_path / f"{paper_id}.pdf"
        if pdf_path.exists():
            return pdf_path
        
        # Try wildcard search (in case filename is slightly different)
        matches = list(self.storage_path.glob(f"*{paper_id}*.pdf"))
        return matches[0] if matches else None
    
    def _extract_text(self, pdf_path: Path) -> tuple[str, List[Dict]]:
        """
        Extract text from PDF using PyMuPDF
        
        Returns:
            (full_text, page_texts)
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        page_texts = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            full_text += text + "\n"
            
            page_texts.append({
                "page": page_num,
                "text": text,
                "char_count": len(text)
            })
        
        doc.close()
        return full_text, page_texts
    
    def _identify_sections(self, full_text: str, page_texts: List[Dict]) -> Dict[str, str]:
        """
        Identify major sections in the paper
        
        Returns:
            Dict mapping section names to their text content
        """
        sections = {}
        
        # Common section headers (case-insensitive)
        section_patterns = {
            'abstract': r'\n\s*(?:abstract|summary)\s*\n',
            'introduction': r'\n\s*(?:1\.?\s*)?introduction\s*\n',
            'related_work': r'\n\s*(?:2\.?\s*)?(?:related work|background|literature review)\s*\n',
            'methodology': r'\n\s*(?:3\.?\s*)?(?:method|methodology|approach|model)\s*\n',
            'experiments': r'\n\s*(?:4\.?\s*)?(?:experiment|evaluation|results)\s*\n',
            'results': r'\n\s*(?:5\.?\s*)?(?:results|findings)\s*\n',
            'discussion': r'\n\s*(?:6\.?\s*)?discussion\s*\n',
            'conclusion': r'\n\s*(?:7\.?\s*)?conclusion\s*\n',
            'references': r'\n\s*references\s*\n'
        }
        
        text_lower = full_text.lower()
        
        # Find section positions
        section_positions = []
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                pos = matches[0].start()
                section_positions.append((pos, section_name))
        
        # Sort by position
        section_positions.sort()
        
        # Extract text for each section
        for i, (pos, name) in enumerate(section_positions):
            # Get text until next section (or end)
            if i < len(section_positions) - 1:
                next_pos = section_positions[i + 1][0]
                section_text = full_text[pos:next_pos]
            else:
                section_text = full_text[pos:]
            
            # Clean up section text
            section_text = section_text.strip()
            
            # Skip if too short or if it's references
            if len(section_text) > 100 and name != 'references':
                sections[name] = section_text
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections['full_text'] = full_text.strip()
        
        return sections
    
    def _create_chunks(self, sections: Dict[str, str], paper_id: str, paper_title: str) -> List[Dict]:
        """
        Create chunks from sections
        
        Strategy:
        - Abstract: Keep whole (usually short)
        - Other sections: Split into ~500-1000 token chunks
        - Preserve paragraph boundaries
        """
        chunks = []
        chunk_id = 0
        
        for section_name, section_text in sections.items():
            # For abstract, keep whole
            if section_name == 'abstract':
                chunks.append({
                    'chunk_id': f"{paper_id}_chunk_{chunk_id}",
                    'paper_id': paper_id,
                    'paper_title': paper_title,
                    'section': section_name,
                    'text': section_text,
                    'char_count': len(section_text),
                    'token_estimate': len(section_text.split()),
                    'chunk_index': chunk_id
                })
                chunk_id += 1
            else:
                # Split long sections into chunks
                section_chunks = self._split_into_chunks(section_text, max_tokens=800)
                
                for chunk_text in section_chunks:
                    chunks.append({
                        'chunk_id': f"{paper_id}_chunk_{chunk_id}",
                        'paper_id': paper_id,
                        'paper_title': paper_title,
                        'section': section_name,
                        'text': chunk_text,
                        'char_count': len(chunk_text),
                        'token_estimate': len(chunk_text.split()),
                        'chunk_index': chunk_id
                    })
                    chunk_id += 1
        
        return chunks
    
    def _split_into_chunks(self, text: str, max_tokens: int = 800) -> List[str]:
        """
        Split text into chunks at paragraph boundaries
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk (approximate)
            
        Returns:
            List of text chunks
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = len(para.split())
            
            # If single paragraph is too long, split it
            if para_tokens > max_tokens:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                
                # Split long paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_count = 0
                
                for sentence in sentences:
                    sent_tokens = len(sentence.split())
                    if temp_count + sent_tokens > max_tokens and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_count = sent_tokens
                    else:
                        temp_chunk.append(sentence)
                        temp_count += sent_tokens
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            
            # If adding paragraph exceeds max, save current chunk
            elif current_token_count + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_token_count = para_tokens
            
            # Otherwise add to current chunk
            else:
                current_chunk.append(para)
                current_token_count += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _save_chunks(self, paper_id: str, chunks: List[Dict]):
        """Save chunks to JSON file"""
        output_path = self.chunks_path / f"{paper_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'paper_id': paper_id,
                'total_chunks': len(chunks),
                'chunks': chunks
            }, f, indent=2, ensure_ascii=False)
    
    def get_chunks(self, paper_id: str) -> Optional[Dict]:
        """Load chunks from file"""
        chunk_file = self.chunks_path / f"{paper_id}.json"
        
        if not chunk_file.exists():
            return None
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def is_processed(self, paper_id: str) -> bool:
        """Check if paper has been processed"""
        return (self.chunks_path / f"{paper_id}.json").exists()


# Test function
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Test with a paper ID
    result = processor.process_paper(
        "s2_test123",
        "Test Paper Title"
    )
    
    print(json.dumps(result, indent=2))