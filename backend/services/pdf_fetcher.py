import os
import requests
from typing import Optional, Dict, Any
from pathlib import Path
import hashlib
from datetime import datetime


class PDFFetcher:
    """Service for downloading and storing PDFs"""
    
    def __init__(self, storage_path: str = "./storage/papers"):
        """
        Initialize PDF fetcher
        
        Args:
            storage_path: Directory to store downloaded PDFs
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (ScholarSync Research Assistant)'
        })
    
    def download_pdf(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download PDF for a paper
        
        Args:
            paper: Paper dictionary with pdf_url
            
        Returns:
            Dict with download status and local path
        """
        result = {
            'success': False,
            'local_path': None,
            'error': None,
            'paywalled': False
        }
        
        pdf_url = paper.get('pdf_url')
        
        if not pdf_url:
            result['error'] = 'No PDF URL available'
            result['paywalled'] = True
            return result
        
        try:
            # Generate unique filename
            paper_id = paper.get('id', hashlib.md5(paper['title'].encode()).hexdigest())
            filename = self._sanitize_filename(f"{paper_id}.pdf")
            local_path = self.storage_path / filename
            
            # Check if already downloaded
            if local_path.exists():
                result['success'] = True
                result['local_path'] = str(local_path)
                result['cached'] = True
                return result
            
            # Download PDF
            print(f"Downloading PDF from: {pdf_url}")
            response = self.session.get(pdf_url, timeout=30, stream=True)
            
            # Check if successful
            if response.status_code == 200:
                # Verify it's actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    result['error'] = f'Not a PDF file (content-type: {content_type})'
                    result['paywalled'] = True
                    return result
                
                # Save PDF
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file size
                file_size = local_path.stat().st_size
                if file_size < 1000:  # Less than 1KB - probably not a real PDF
                    local_path.unlink()  # Delete invalid file
                    result['error'] = 'Downloaded file too small (likely paywall page)'
                    result['paywalled'] = True
                    return result
                
                result['success'] = True
                result['local_path'] = str(local_path)
                result['file_size'] = file_size
                result['cached'] = False
                
                print(f"✓ PDF saved: {local_path} ({file_size / 1024:.1f} KB)")
                
            elif response.status_code == 403 or response.status_code == 401:
                result['error'] = 'Access denied (paywalled)'
                result['paywalled'] = True
                
            else:
                result['error'] = f'HTTP {response.status_code}'
                result['paywalled'] = True
                
        except requests.exceptions.Timeout:
            result['error'] = 'Download timeout'
            
        except requests.exceptions.ConnectionError:
            result['error'] = 'Connection failed'
            
        except Exception as e:
            result['error'] = f'Download failed: {str(e)}'
        
        return result
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be filesystem-safe"""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:196] + ext
        
        return filename
    
    def get_local_path(self, paper_id: str) -> Optional[str]:
        """Get local path for a paper if it exists"""
        filename = self._sanitize_filename(f"{paper_id}.pdf")
        local_path = self.storage_path / filename
        
        return str(local_path) if local_path.exists() else None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored PDFs"""
        pdf_files = list(self.storage_path.glob("*.pdf"))
        
        total_size = sum(f.stat().st_size for f in pdf_files)
        
        return {
            'total_papers': len(pdf_files),
            'total_size_mb': total_size / (1024 * 1024),
            'storage_path': str(self.storage_path)
        }


# Test function
if __name__ == "__main__":
    fetcher = PDFFetcher()
    
    # Test with a sample paper
    test_paper = {
        'id': 'test_paper',
        'title': 'Test Paper',
        'pdf_url': 'https://arxiv.org/pdf/2301.00001.pdf'
    }
    
    result = fetcher.download_pdf(test_paper)
    print(result)
    
    # Get stats
    stats = fetcher.get_storage_stats()
    print(f"\nStorage stats: {stats}")