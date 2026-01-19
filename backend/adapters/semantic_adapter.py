import requests
from typing import List, Dict, Any
import time

class SemanticScholarAdapter:
    """Adapter for Semantic Scholar API"""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.session = requests.Session()
        # No API key needed for basic usage (rate limited to 100 requests/5min)
        # Nice UA so they know this is a small student project, not a botnet :)
        self.session.headers.update({
            "User-Agent": "ScholarSync/0.7 (student project)",
            "Accept": "application/json",
        })
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        try:
            # Build query parameters
            params = {
                'query': query,
                'limit': max_results,
                'fields': 'paperId,title,abstract,year,authors,citationCount,publicationDate,url,externalIds,fieldsOfStudy'
            }
            
            # Make request
            url = f"{self.BASE_URL}/paper/search"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 429:
                print("Semantic Scholar rate-limited (HTTP 429). Skipping this source for now.")
                return []
                # time.sleep(5)
                # response = self.session.get(url, params=params, timeout=10)
            
            # response.raise_for_status()

            if not response.ok:
                print(
                    f"Semantic Scholar HTTP {response.status_code}: "
                    f"{response.text[:200]!r}"
                )
                return []

            # Make sure it's JSON
            if "application/json" not in response.headers.get("Content-Type", ""):
                print("Semantic Scholar returned non-JSON response. Skipping this source.")
                return []
            
            try:
                data = response.json()
            except ValueError as e:
                print(f"Semantic Scholar JSON parse error: {e}")
                return []

            papers = self._parse_response(data)
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {str(e)}")
            return []
    
    def _parse_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar JSON response"""
        papers = []
        
        try:
            for item in data.get('data', []):
                try:
                    paper = self._extract_paper_data(item)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"Error parsing paper: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Response parsing error: {str(e)}")
        
        return papers
    
    def _extract_paper_data(self, item: Dict) -> Dict[str, Any]:
        """Extract paper data from API response"""
        
        # Extract basic info
        paper_id = item.get('paperId', '')
        title = item.get('title', 'No title')
        abstract = item.get('abstract', 'No abstract available')
        year = item.get('year')
        citation_count = item.get('citationCount', 0)
        
        # Extract authors
        authors = []
        for author in item.get('authors', []):
            authors.append(author.get('name', 'Unknown'))
        
        # Extract external IDs (DOI, arXiv, etc.)
        external_ids = item.get('externalIds', {})
        doi = external_ids.get('DOI')
        arxiv_id = external_ids.get('ArXiv')
        
        # Build PDF URL if available
        pdf_url = None
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Extract fields of study
        fields = item.get('fieldsOfStudy', [])
        
        return {
            'id': f"s2_{paper_id}",
            'source': 'semantic_scholar',
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'year': year,
            'published_date': item.get('publicationDate'),
            'pdf_url': pdf_url,
            'citation_count': citation_count,
            'doi': doi,
            'arxiv_id': arxiv_id,
            'url': item.get('url', f"https://www.semanticscholar.org/paper/{paper_id}"),
            'fields_of_study': fields
        }



# Test function
if __name__ == "__main__":
    adapter = SemanticScholarAdapter()
    results = adapter.search("audio event detection", max_results=3)
    
    print(f"\nFound {len(results)} papers:\n")
    for paper in results:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"Year: {paper['year']}")
        print(f"Citations: {paper['citation_count']}")
        print(f"DOI: {paper['doi']}\n")