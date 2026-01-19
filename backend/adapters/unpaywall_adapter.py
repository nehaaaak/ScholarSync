import requests
from typing import Optional, Dict, Any

class UnpaywallAdapter:
    """Adapter for Unpaywall API to find free PDFs"""
    
    BASE_URL = "https://api.unpaywall.org/v2"
    
    def __init__(self, email: str = "scholarsync@example.com"):
        """
        Initialize Unpaywall adapter
        
        Args:
            email: Email for Unpaywall API (required by their terms)
        """
        self.email = email
        self.session = requests.Session()
    
    def get_free_pdf(self, doi: str) -> Optional[str]:
        """
        Get free PDF URL for a paper by DOI
        
        Args:
            doi: Paper DOI
            
        Returns:
            Free PDF URL if available, None otherwise
        """
        if not doi:
            return None
        
        try:
            url = f"{self.BASE_URL}/{doi}"
            params = {'email': self.email}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._extract_pdf_url(data)
            
            return None
            
        except Exception as e:
            print(f"Unpaywall error for DOI {doi}: {str(e)}")
            return None
    
    def _extract_pdf_url(self, data: Dict) -> Optional[str]:
        """Extract best available PDF URL from Unpaywall response"""
        
        # Check if it's open access
        if not data.get('is_oa'):
            return None
        
        # Try to get best OA location
        best_oa = data.get('best_oa_location')
        if best_oa and best_oa.get('url_for_pdf'):
            return best_oa['url_for_pdf']
        
        # Fall back to first OA location
        oa_locations = data.get('oa_locations', [])
        for location in oa_locations:
            if location.get('url_for_pdf'):
                return location['url_for_pdf']
        
        return None
    
    def enrich_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a paper dict with free PDF URL if available
        
        Args:
            paper: Paper dictionary with 'doi' field
            
        Returns:
            Paper dict with updated pdf_url if found
        """
        doi = paper.get('doi')
        
        if doi and not paper.get('pdf_url'):
            free_pdf = self.get_free_pdf(doi)
            if free_pdf:
                paper['pdf_url'] = free_pdf
                paper['open_access'] = True
        
        return paper



# Test function
if __name__ == "__main__":
    adapter = UnpaywallAdapter()
    
    # Test with a known OA paper
    test_doi = "10.1371/journal.pone.0000308"
    pdf_url = adapter.get_free_pdf(test_doi)
    
    print(f"DOI: {test_doi}")
    print(f"Free PDF: {pdf_url}")