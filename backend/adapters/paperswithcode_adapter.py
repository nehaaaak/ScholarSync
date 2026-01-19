import requests
from typing import List, Dict, Any, Optional

class PapersWithCodeAdapter:
    """Adapter for PapersWithCode API"""
    
    BASE_URL = "https://paperswithcode.com/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScholarSync/0.7 (student project)",
            "Accept": "application/json",
        })
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search PapersWithCode for papers
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries with code repos
        """
        try:
            # Search papers endpoint
            url = f"{self.BASE_URL}/papers/"
            params = {
                'q': query,
                'items_per_page': max_results
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"PapersWithCode API returned {response.status_code}")
                return []
            
            # Some rate-limit / error pages are HTML but with 200 status
            if "application/json" not in response.headers.get("Content-Type", ""):
                print("PapersWithCode returned non-JSON response. Skipping this source.")
                return []
            
            try:
                data = response.json()
            except ValueError as e:
                print(f"PapersWithCode JSON parse error: {e}")
                return []
            
            papers = self._parse_response(data)           
            return papers
            
        except Exception as e:
            print(f"PapersWithCode search error: {str(e)}")
            return []
    
    def get_paper_repos(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get code repositories for a specific paper"""
        try:
            url = f"{self.BASE_URL}/papers/{paper_id}/repositories/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            return []
            
        except Exception as e:
            print(f"Error fetching repos: {str(e)}")
            return []
    
    def _parse_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse PapersWithCode API response"""
        papers = []
        
        try:
            for item in data.get('results', []):
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
        
        paper_id = item.get('id', '')
        title = item.get('title', 'No title')
        abstract = item.get('abstract', '')
        
        # Extract paper URL
        paper_url = item.get('url_abs') or item.get('url_pdf', '')
        
        # Check for arXiv ID
        arxiv_id = item.get('arxiv_id')
        
        # Get repository information
        repos = []
        if 'repository_count' in item and item['repository_count'] > 0:
            # Fetch actual repos
            repos = self.get_paper_repos(paper_id)
        
        # Build code repo links
        code_urls = []
        for repo in repos[:3]:  # Top 3 repos
            if repo.get('url'):
                code_urls.append({
                    'url': repo['url'],
                    'stars': repo.get('stars', 0),
                    'framework': repo.get('framework', 'Unknown')
                })
        
        # Extract year from published date if available
        year = None
        if 'published' in item:
            try:
                year = int(item['published'][:4])
            except:
                pass
        
        # PDF URL
        pdf_url = None
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif item.get('url_pdf'):
            pdf_url = item['url_pdf']
        
        return {
            'id': f"pwc_{paper_id}",
            'source': 'paperswithcode',
            'title': title,
            'authors': [],  # PapersWithCode doesn't always provide authors
            'abstract': abstract,
            'year': year,
            'pdf_url': pdf_url,
            'arxiv_id': arxiv_id,
            'url': f"https://paperswithcode.com/paper/{paper_id}",
            'code_urls': code_urls,
            'has_code': len(code_urls) > 0,
            'citation_count': None
        }



# Test function
if __name__ == "__main__":
    adapter = PapersWithCodeAdapter()
    results = adapter.search("audio event detection", max_results=3)
    
    print(f"\nFound {len(results)} papers:\n")
    for paper in results:
        print(f"Title: {paper['title']}")
        print(f"Has code: {paper['has_code']}")
        if paper['code_urls']:
            print(f"Repos: {len(paper['code_urls'])}")
        print()