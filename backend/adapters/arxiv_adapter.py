import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from datetime import datetime
import urllib.parse


class ArxivAdapter:
    """Adapter for arXiv API"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.session = requests.Session()
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries
        """
        try:
            # Build query parameters
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results * 2,  # Get extra for filtering
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            # Make request
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_response(response.text)
            
            # Limit to max_results
            return papers[:max_results]
            
        except Exception as e:
            print(f"arXiv search error: {str(e)}")
            return []
    
    def _parse_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse arXiv XML response"""
        papers = []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_text)
            
            # Namespace for arXiv
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Extract entries
            for entry in root.findall('atom:entry', ns):
                try:
                    paper = self._extract_paper_data(entry, ns)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"Error parsing entry: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"XML parsing error: {str(e)}")
        
        return papers
    
    def _extract_paper_data(self, entry, ns) -> Dict[str, Any]:
        """Extract paper data from XML entry"""
        
        # Extract ID
        arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
        
        # Extract title
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        
        # Extract abstract
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        
        # Extract published date
        published = entry.find('atom:published', ns).text
        year = datetime.fromisoformat(published.replace('Z', '+00:00')).year
        
        # Extract PDF link
        pdf_link = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_link = link.get('href')
                break
        
        # If no PDF link found, construct it
        if not pdf_link:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Extract categories
        categories = []
        primary_category = entry.find('arxiv:primary_category', ns)
        if primary_category is not None:
            categories.append(primary_category.get('term'))
        
        return {
            'id': f"arxiv_{arxiv_id}",
            'source': 'arxiv',
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'year': year,
            'published_date': published,
            'pdf_url': pdf_link,
            'arxiv_id': arxiv_id,
            'categories': categories,
            'url': f"https://arxiv.org/abs/{arxiv_id}",
            'citation_count': None,  # arXiv doesn't provide this
            'doi': None
        }


# Test function
if __name__ == "__main__":
    adapter = ArxivAdapter()
    results = adapter.search("audio event detection", max_results=3)
    
    print(f"\nFound {len(results)} papers:\n")
    for paper in results:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"Year: {paper['year']}")
        print(f"PDF: {paper['pdf_url']}\n")