import streamlit as st
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ScholarSync",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend URL
BACKEND_URL = "http://localhost:8000"


@st.dialog("💬 Chat about this paper", width="large")
def paper_chat_dialog(paper):
    paper_id = paper["id"]
    paper_title = paper["title"]

    if paper_id not in st.session_state.chat_history:
        st.session_state.chat_history[paper_id] = []

    history = st.session_state.chat_history[paper_id]

    st.caption(paper_title)

    # Chat history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_query = st.chat_input("Ask a question about this paper")

    if user_query:
        # Show user message immediately
        history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                abstract = paper.get("abstract", "")
                metadata = {
                    "year": paper.get("year"),
                    "authors": paper.get("authors", []),
                }

                embed_abstract_if_needed(
                    paper["id"],
                    paper["title"],
                    abstract,
                    metadata,
                )

                response = ask_paper_api(
                    paper["id"],
                    user_query,
                    paper["title"],
                    metadata,
                )

                answer = response.get("answer", "Something went wrong.") if response else "Something went wrong."
                st.markdown(answer)

        history.append({"role": "assistant", "content": answer})


@st.dialog("🌐 Research Assistant", width="large")
def global_chat_dialog():
    history = st.session_state.global_chat_history

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask across all papers")

    if user_query:
        history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching papers..."):
                response = ask_global_api(user_query)
                answer = response["answer"] if response else "No answer found."
                st.markdown(answer)

        history.append({"role": "assistant", "content": answer})



def init_session_state():
    """Initialize all session state variables"""

    # Search-related
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_max_results" not in st.session_state:
        st.session_state.last_max_results = 5
    if "last_search" not in st.session_state:
        st.session_state.last_search = ""
    if "last_sort" not in st.session_state:
        st.session_state.last_sort = "relevance"

    # Download tracking
    if "download_status" not in st.session_state:
        st.session_state.download_status = {}

    # Bookmarks
    if "bookmarks" not in st.session_state:
        st.session_state.bookmarks = []
    if "bookmarked_papers" not in st.session_state:
        st.session_state.bookmarked_papers = {}

    # Page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "search"  

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}    
    if 'global_chat_history' not in st.session_state:
        st.session_state.global_chat_history = []    
    # if 'show_paper_chat' not in st.session_state:
    #     st.session_state.show_paper_chat = False    
    # if 'active_chat_paper_id' not in st.session_state:
    #     st.session_state.active_chat_paper_id = None    
    # if 'active_chat_paper_title' not in st.session_state:
    #     st.session_state.active_chat_paper_title = None    
    # if 'active_chat_paper' not in st.session_state:
    #     st.session_state.active_chat_paper = None
    if 'show_global_chat' not in st.session_state:
        st.session_state.show_global_chat = False


def save_state_to_url():
    """Save current state to URL parameters"""
    if st.session_state.last_query:
        st.query_params.update(
            q=st.session_state.last_query,
            n=st.session_state.last_max_results,
            sort=st.session_state.last_sort,
        )

def restore_state_from_url():
    params = st.query_params
    
    q_list = params.get("q")
    if q_list and not st.session_state.search_results:
        query = q_list[0]

        n_list = params.get("n", ["5"])
        try:
            max_results = int(n_list[0])
        except ValueError:
            max_results = 5

        # restore into session state
        st.session_state.last_query = query
        st.session_state.last_max_results = max_results

        return query, max_results

    return None, None

# Initialize on app load
init_session_state()

url_query, url_max = restore_state_from_url()

def search_papers(query: str, max_results: int = 5, sort_by: str = "relevance", offset: int = 0):
    """Search for papers via backend"""
    try:
        payload = {
            "query": query,
            "max_results": max_results,
            "include_code": True,
            "sort_by": sort_by,  # new
            "offset": offset
        }
        response = requests.post(
            f"{BACKEND_URL}/search",
            json=payload,
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None


def download_paper(paper_id: str, paper_title: str, pdf_url: str):
    """Download a paper PDF"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/get-paper",
            json={
                "paper_id": paper_id,
                "paper_title": paper_title,
                "pdf_url": pdf_url
            },
            timeout=60
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_storage_stats():
    """Get storage statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/storage-stats", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None
    

def embed_abstract_if_needed(paper_id: str, paper_title: str, abstract: str, metadata: dict) -> bool:
    """Embed abstract if not already embedded."""
    try:
        # 1) Check embedding status
        status_response = requests.get(
            f"{BACKEND_URL}/embedding-status/{paper_id}",
            timeout=5,
        )
        if status_response.status_code == 200:
            status = status_response.json()
            if status.get("has_abstract"):
                # Already embedded → nothing to do
                return True

        # 2) Call /embed-abstract
        payload = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "abstract": abstract,
            "paper_metadata": metadata,
        }

        response = requests.post(
            f"{BACKEND_URL}/embed-abstract",
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            return False

        data = response.json()
        return data.get("success", False)

    except Exception as e:
        st.error(f"Failed to embed abstract: {e}")
        return False
    

def ask_paper_api(paper_id: str, question: str, paper_title: str, metadata: dict):
    """Ask a question about a specific paper (with RAG answer)"""
    try:
        payload = {
            "paper_id": paper_id,
            "question": question,
            "paper_title": paper_title,
            "paper_metadata": metadata
        }
        
        response = requests.post(
            f"{BACKEND_URL}/ask-paper",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except Exception as e:
        st.error(f"Q&A failed: {e}")
        return None


def ask_global_api(question: str, filters: dict = None, max_papers: int = 5):
    """Ask a question across all papers (with RAG synthesis)"""
    try:
        payload = {
            "question": question,
            "filters": filters,
            "max_papers": max_papers
        }
        
        response = requests.post(
            f"{BACKEND_URL}/ask-global",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except Exception as e:
        st.error(f"Global Q&A failed: {e}")
        return None


def search_within_paper_api(paper_id: str, query: str, top_k: int = 6):
    """Search within a specific paper (per-paper semantic search)."""
    try:
        payload = {
            "paper_id": paper_id,
            "query": query,
            "top_k": top_k,
        }

        response = requests.post(
            f"{BACKEND_URL}/search-within-paper",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()   # { paper_id, query, results, total_found }
        else:
            st.error(f"Search failed: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Search failed: {e}")
        return None


def search_global_api(query: str, top_k: int = 15, filters: dict = None):
    """Search across all embedded papers (global semantic search)."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
        }
        if filters is not None:
            payload["filters"] = filters  

        response = requests.post(
            f"{BACKEND_URL}/search-global",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()   # { query, results, total_chunks, papers_cited, total_papers }
        else:
            st.error(f"Global search failed: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Global search failed: {e}")
        return None
    

def summarize_to_notion_api(paper_id: str, paper_title: str):
    """Trigger LangFlow summarization to Notion"""
    try:
        payload = {
            "paper_id": paper_id,
            "paper_title": paper_title
        }
        
        response = requests.post(
            f"{BACKEND_URL}/summarize-to-notion",
            json=payload,
            timeout=180  #3 minutes
        )
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except requests.exceptions.Timeout:
        st.warning("The request timed out, but the process might still be running in the background.")
        return None
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return None
    


# Header
st.title("📚 ScholarSync")
st.caption("Your AI-Powered Research Assistant!")

# Check backend
try:
    health = requests.get(f"{BACKEND_URL}/status", timeout=3).json()
except:
    st.error("❌ Backend Offline")
    st.stop()

st.divider()


# Sidebar
with st.sidebar:
    # Navigation
    page = st.radio("View", ["Search", "Bookmarks"], index=0)
    
    st.header("🔄 Session")
    
    # Show current session state
    if st.session_state.search_results:
        st.success(f"✓ '{st.session_state.last_query}'")
        st.caption(f"{len(st.session_state.search_results.get('papers', []))} papers loaded")
    
    if st.session_state.bookmarks:
        st.info(f"⭐ {len(st.session_state.bookmarks)} bookmarked")
    
    # Clear session button
    if st.button("🗑️ Clear Session", use_container_width=True):
        for key in ['search_results', 'bookmarks', 'bookmarked_papers', 'last_query']:
            st.session_state[key] = [] if 'bookmarks' in key else None if 'results' in key else {}
        st.rerun()
    
    st.divider()

    st.header("📊 Quick Stats")
    
    # Storage stats
    stats = get_storage_stats()
    if stats:
        st.subheader("💾 Local Storage")
        st.metric("Papers Stored", stats['total_papers'])
        st.metric("Storage Used", f"{stats['total_size_mb']:.1f} MB")
    
    st.divider()
    
    # Example queries
    st.subheader("💡 Example Queries")
    example_queries = [
        "audio event detection",
        "sound classification deep learning",
        "acoustic scene analysis",
        "speech recognition transformers",
        "music information retrieval"
    ]
    
    for example in example_queries:
        if st.button(f"🔍 {example}", key=f"ex_{example}", use_container_width=True):
            st.session_state.search_input = example
            st.rerun()
    
    st.divider()

    st.subheader("🎯 Chat Stats")

    # Count total messages
    total_per_paper = sum(len(msgs) for msgs in st.session_state.chat_history.values())
    total_global = len(st.session_state.global_chat_history)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Per-Paper", total_per_paper)
    with col2:
        st.metric("Global", total_global)

    if total_per_paper > 0 or total_global > 0:
        if st.button("🗑️ Clear All Chats", key="clear_all_chats"):
            st.session_state.chat_history = {}
            st.session_state.global_chat_history = []
            st.success("✅ All chats cleared!")
            st.rerun()



if page == "Search":
    # Search Interface
    st.subheader("🔍 Search Research Papers")

    if url_query and not st.session_state.search_results:
        with st.spinner(f"🔍 Restoring search for '{url_query}'..."):
            try:
                st.session_state.search_results = search_papers(
                    url_query,
                    url_max or st.session_state.last_max_results,
                    sort_by=st.session_state.last_sort,
                )
                st.session_state.last_query = url_query
            except Exception as e:
                st.error(f"❌ Failed to restore search: {str(e)}")

    # Sorting options mapping
    sort_label_to_backend = {
        "Relevance (Recommended)": "relevance",
        "Most Recent": "most_recent",
        "Most Cited": "most_cited",
        "Has Code": "has_code",
    }
    sort_labels = list(sort_label_to_backend.keys())

    # Pick default index based on last_sort in session
    default_sort_idx = 0
    for i, lbl in enumerate(sort_labels):
        if sort_label_to_backend[lbl] == st.session_state.last_sort:
            default_sort_idx = i
            break

    # Single row: query | max_results | sort
    col_query, col_max, col_sort = st.columns([6, 1, 2])

    with col_query:
        # st.caption("Search query")
        query = st.text_input(
            "Search query",
            value=st.session_state.last_query,
            placeholder="e.g., audio event detection, sound classification, etc.",
            label_visibility="collapsed",
            key="search_input"
        )

    with col_max:
        # small label above number input
        # st.caption("Max results")
        max_results = st.number_input(
            "Max results",
            min_value=1,
            max_value=10,
            value=st.session_state.last_max_results,
            label_visibility="collapsed"
        )

    with col_sort:
        # st.caption("Sort by")
        sort_choice = st.selectbox(
            "Sort by",
            sort_labels,
            index=default_sort_idx,
            label_visibility="collapsed",
        )

    selected_sort_by = sort_label_to_backend[sort_choice]

    # Search button (full width row below)
    search_button = st.button("🔎 Search", type="primary", use_container_width=True)

    # # Detect Enter key press by checking if query changed
    enter_pressed = False
    if query and query != st.session_state.last_search and len(query.strip()) >= 2:
        enter_pressed = True
        st.session_state.last_search = query

    # Handle search (both button click AND Enter key)
    search_triggered = search_button or enter_pressed

    if search_triggered:
        if not query or len(query.strip()) == 0:
            st.warning("⚠️ Please enter a search query")
        elif len(query.strip()) < 2:
            st.warning("⚠️ Please enter at least 2 characters")
        else:
            clean_query = query.strip()
            with st.spinner(f"🔍 Searching for '{clean_query}'..."):
                try:
                    st.session_state.search_results = search_papers(clean_query, max_results, sort_by=selected_sort_by)
                    st.session_state.download_status = {}
                    st.session_state.last_search = query

                    st.session_state.last_query = clean_query
                    st.session_state.last_max_results = max_results
                    st.session_state.last_sort = selected_sort_by

                    save_state_to_url()

                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.session_state.search_results = None

    # Display results
    if st.session_state.search_results:
        results = st.session_state.search_results
        
        if results.get('papers') and len(results['papers']) > 0:
            # Show sources (only non-zero)
            source_labels = {
                "arxiv": "arXiv",
                "semantic_scholar": "Semantic Scholar",
                "paperswithcode": "PapersWithCode",
            }

            source_counts = {}
            for p in results["papers"]:
                src = p.get("source")
                if not src:
                    continue
                source_counts[src] = source_counts.get(src, 0) + 1

            nonzero_sources = [
                (key, count)
                for key, count in results.get("sources", {}).items()
                if count and count > 0
            ]

            if nonzero_sources:
                cols = st.columns(len(nonzero_sources))
                for col, (key, count) in zip(cols, nonzero_sources):
                    with col:
                        st.metric(source_labels.get(key, key), count)
    
            st.divider()
            
            # Display papers
            for idx, paper in enumerate(results['papers'], 1):
                # Create a stable container for each paper
                paper_container = st.container()
                
                with paper_container:
                    header_col1, header_col2 = st.columns([4, 1])
                    
                    with header_col1:
                        st.markdown(f"### {idx}. {paper['title']}")
                        
                        if paper.get('authors'):
                            authors = ', '.join(paper['authors'][:3])
                            if len(paper['authors']) > 3:
                                authors += f" et al. ({len(paper['authors'])} authors)"
                            st.caption(f"👥 {authors}")
                        
                        # Metadata
                        meta_parts = []
                        
                        if paper.get('year'):
                            meta_parts.append(f"📅 {paper['year']}")
                        
                        source_icons = {
                            'arxiv': '🔬 arXiv',
                            'semantic_scholar': '📊 Semantic Scholar',
                            'paperswithcode': '💻 PapersWithCode'
                        }
                        meta_parts.append(source_icons.get(paper.get('source'), paper.get('source', 'Unknown')))
                        
                        if paper.get('has_code'):
                            meta_parts.append('💻 Code Available')
                        
                        if paper.get('open_access'):
                            meta_parts.append('🔓 Open Access')
                        
                        # Check download status
                        paper_key = f"{paper['id']}_{idx}"
                        if st.session_state.download_status.get(paper_key, False) or paper.get('downloaded'):
                            meta_parts.append('✅ Downloaded')
                        
                        st.caption(' | '.join(meta_parts))

                        # Bookmark toggle
                        is_bookmarked = paper["id"] in st.session_state.bookmarks
                        bm_label = "⭐ Bookmarked" if is_bookmarked else "☆ Bookmark"

                        if st.button(
                            bm_label,
                            key=f"bm_{paper['id']}_{idx}",
                        ):
                            if is_bookmarked:
                                # Remove from list and dict
                                st.session_state.bookmarks.remove(paper["id"])
                                st.session_state.bookmarked_papers.pop(paper["id"], None)
                            else:
                                # Add to list and dict
                                st.session_state.bookmarks.append(paper["id"])
                                st.session_state.bookmarked_papers[paper["id"]] = paper

                            st.rerun()
                    
                    with header_col2:
                        citation_count = paper.get('citation_count')
                        if citation_count is not None and citation_count > 0:
                            st.metric("Citations", citation_count)
                        elif citation_count == 0:
                            st.caption("No citations yet")
                    
                    # Abstract
                    abstract = paper.get('abstract')
                    if abstract and isinstance(abstract, str):
                        abstract_preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                    else:
                        abstract_preview = 'No abstract available'
                    st.write(abstract_preview)
                    
                    # Action buttons
                    has_pdf = paper.get('pdf_url') is not None
                    has_url = paper.get('url') is not None
                    has_code = paper.get('code_urls') and len(paper['code_urls']) > 0
                    
                    btn_cols = st.columns(6)
                    col_idx = 0
                    
                    # PDF or View button
                    with btn_cols[col_idx]:
                        if has_pdf:
                            st.link_button("📄 PDF", paper['pdf_url'], use_container_width=True)
                            col_idx += 1
                        elif has_url:
                            st.link_button("🔗 View", paper['url'], use_container_width=True)
                            col_idx += 1
                    
                    # View button
                    if has_pdf and has_url:
                        with btn_cols[col_idx]:
                            st.link_button("🔗 View", paper['url'], use_container_width=True)
                            col_idx += 1
                    
                    # Code button
                    if has_code:
                        with btn_cols[col_idx]:
                            st.link_button("💻 Code", paper['code_urls'][0]['url'], use_container_width=True)
                            col_idx += 1
                    

                    # Download button
                    if has_pdf:
                        with btn_cols[col_idx]:
                            download_key = f"dl_{paper['id']}_{idx}"
                            paper_key = f"{paper['id']}_{idx}"
                            
                            # is_downloaded = st.session_state.download_status.get(paper_key, False) or paper.get('downloaded')
 
                            is_downloaded = paper.get("downloaded", False)
                            
                            if is_downloaded:
                                st.button("✅ Saved", key=download_key, disabled=True, use_container_width=True)
                            else:
                                status_placeholder = st.empty()
                                if st.button("💾 Download", key=download_key, use_container_width=True):
                                    # Show spinner BELOW button (cleaner look)
                                    with status_placeholder:
                                        with st.spinner("⏳ Downloading PDF..."):
                                            result = download_paper(
                                                paper['id'],
                                                paper['title'],
                                                paper['pdf_url']
                                            )
                                    
                                    if result and result.get('success'):
                                        # Mark as downloaded
                                        st.session_state.download_status[paper_key] = True
                                        paper['downloaded'] = True
                                        if result.get("local_path"):
                                            paper["local_path"] = result["local_path"]
                                        
                                        # Show toast notification
                                        file_size_kb = result.get('file_size', 0) / 1024
                                        st.toast(f"Downloaded! ({file_size_kb:.0f} KB)", icon="✅")
                                        
                                        # Wait 1 seconds (user sees toast, feels completed)
                                        import time
                                        time.sleep(1)
                                        
                                        # Then rerun (flash happens AFTER user already happy)
                                        # st.rerun()
                                        
                                    elif result and result.get('paywalled'):
                                        st.toast("⚠️ Paper is paywalled", icon="⚠️")
                                        
                                    else:
                                        error_msg = result.get('error', 'Unknown') if result else 'Failed'
                                        st.toast(f"❌ {error_msg}", icon="❌")
                            
                            col_idx += 1


                    # Summarize button
                    if has_pdf or paper.get('downloaded'):
                        with btn_cols[col_idx]:
                            if st.button("🤖 Summarize", key=f"sum_{idx}", use_container_width=True):
                                with st.spinner("🤖 Generating summary and saving to Notion..."):
                                    result = summarize_to_notion_api(paper['id'], paper['title'])
                                
                                if result and result.get('success'):
                                    st.success("✅ Summary saved to Notion!")
                                    
                                    notion_url = result.get('notion_url')
                                    if notion_url:
                                        st.link_button("📖 View in Notion", notion_url)
                                    else:
                                        st.info("Check your Notion database!")
                                else:
                                    st.error("❌ Failed to generate summary")
                            
                            col_idx += 1

                    # Ask button (NEW!)
                    if has_pdf or paper.get('downloaded'):
                        with btn_cols[col_idx]:
                            # if st.button("💬 Ask", key=f"ask_{paper['id']}_{idx}", use_container_width=True):
                            #     st.session_state.show_paper_chat = True
                            #     st.session_state.active_chat_paper_id = paper['id']
                            #     st.session_state.active_chat_paper_title = paper['title']
                            #     st.session_state.active_chat_paper = paper

                            #     # 🔒 Close global chat if it was open
                            #     st.session_state.show_global_chat = False

                            #     st.rerun()
                            if st.button("💬 Ask", key=f"ask_{paper['id']}_{idx}", use_container_width=True):
                                paper_chat_dialog(paper)

                    
                    st.divider()

            # Show total available
            total_available = results.get('total_available', len(results['papers']))    
            current_shown = len(results['papers'])
            has_more = results.get('has_more', False)

            load_more_placeholder = st.empty()

            with load_more_placeholder.container():     
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    if has_more:
                        # st.info(f"Showing {current_shown} of {total_available} papers")
                        load_more_clicked = st.button(
                            "📥 Load More Papers",
                            key="load_more",
                            type="secondary",
                            use_container_width=True,
                        )
                    else:
                        st.success(f"✨ All {total_available} papers loaded!")
                        load_more_clicked = False

            if has_more and load_more_clicked:
                # Replace that whole block with just a spinner while loading
                with load_more_placeholder:
                    with st.spinner("Loading more papers..."):
                        current_offset = results.get("offset", 0)
                        new_offset = current_offset + current_shown

                        more_results = search_papers(
                            st.session_state.last_query,
                            max_results=5,  # or st.session_state.last_max_results
                            sort_by=st.session_state.get("last_sort", "relevance"),
                            offset=new_offset,
                        )

                    if more_results and more_results.get("papers"):
                        existing_ids = {p["id"] for p in st.session_state.search_results["papers"]}
                        unique_new_papers = []
                        for p in more_results["papers"]:
                            if p["id"] not in existing_ids:
                                unique_new_papers.append(p)
                                existing_ids.add(p["id"])

                        st.session_state.search_results["papers"].extend(unique_new_papers)

                        # total_found should track what you’re actually showing
                        st.session_state.search_results["total_found"] = len(
                            st.session_state.search_results["papers"]
                        )

                        st.session_state.search_results["has_more"] = more_results.get("has_more", False)
                        st.session_state.search_results["offset"] = new_offset

                        # Merge per-source counts so the header numbers keep increasing correctly
                        current_sources = st.session_state.search_results.get("sources", {}).copy()
                        new_sources = more_results.get("sources", {})

                        for src, count in new_sources.items():
                            current_sources[src] = current_sources.get(src, 0) + count

                        st.session_state.search_results["sources"] = current_sources

                load_more_placeholder.empty()
                # After updating state, rerun with updated papers
                st.rerun()

            st.markdown("<br><br>", unsafe_allow_html=True)

            # --- Bottom-right Global Research Assistant toggle (inside Search page) ---
            spacer_col, btn_col = st.columns([4, 1])
            with btn_col:
                if st.session_state.show_global_chat:
                    if st.button("✖ Close Assistant", key="close_global_bottom", use_container_width=True):
                        st.session_state.show_global_chat = False
                        st.rerun()
                else:
                    # if st.button("🌐 Chat", key="open_global_bottom", use_container_width=True):
                    #     st.session_state.show_global_chat = True

                    #     # 🔒 Close per-paper chat when opening global
                    #     st.session_state.show_paper_chat = False
                    #     st.session_state.active_chat_paper_id = None
                    #     st.session_state.active_chat_paper_title = ""
                    #     st.session_state.active_chat_paper = None

                    #     st.rerun()
                    if st.button("🌐 Open Research Assistant", use_container_width=True):
                        global_chat_dialog()
                    
        elif search_button and query:
            st.warning("🔎 No results found. Try a different query.")



if page == "Bookmarks":
    st.subheader("⭐ Bookmarked Papers")

    if not st.session_state.bookmarks:
        st.info("No bookmarked papers yet. Go to the Search tab and bookmark some papers.")
    else:
        for idx, paper_id in enumerate(st.session_state.bookmarks, 1):
            paper = st.session_state.bookmarked_papers.get(paper_id)
            if not paper:
                continue  # safety

            with st.container():
                st.markdown(f"### {idx}. {paper['title']}")

                if paper.get("authors"):
                    authors = ", ".join(paper["authors"][:3])
                    if len(paper["authors"]) > 3:
                        authors += f" et al. ({len(paper['authors'])} authors)"
                    st.caption(f"👥 {authors}")

                meta_parts = []
                if paper.get("year"):
                    meta_parts.append(f"📅 {paper['year']}")

                source_icons = {
                    "arxiv": "🔬 arXiv",
                    "semantic_scholar": "📊 Semantic Scholar",
                    "paperswithcode": "💻 PapersWithCode",
                }
                meta_parts.append(source_icons.get(paper.get("source"), paper.get("source", "Unknown")))
                st.caption(" | ".join(meta_parts))

                abstract = paper.get("abstract")
                if abstract and isinstance(abstract, str):
                    abstract_preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                else:
                    abstract_preview = "No abstract available"
                st.write(abstract_preview)

                # Quick actions
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if paper.get("pdf_url"):
                        st.link_button("📄 PDF", paper["pdf_url"], use_container_width=True)
                with btn_cols[1]:
                    if paper.get("url"):
                        st.link_button("🔗 View", paper["url"], use_container_width=True)
                with btn_cols[2]:
                    # Allow unbookmark from here
                    if st.button("🗑 Remove", key=f"rm_{paper_id}"):
                        if paper_id in st.session_state.bookmarks:
                            st.session_state.bookmarks.remove(paper_id)
                        st.session_state.bookmarked_papers.pop(paper_id, None)
                        st.rerun()

                st.divider()



st.divider()

# Footer
# st.caption("ScholarSync | Made with ❤️.")

# Custom CSS to inject a fixed footer at the bottom
footer_css = """
<style>
.footer {
    width: 100%;
    background-color: transparent;
    color: grey;
    text-align: center;
    padding: 20px 0px 10px 0px;
    font-size: 14px;
    border-top: 1px solid rgba(151, 151, 151, 0.2);
    margin-top: 50px;
}
</style>
<div class="footer">
    <p>ScholarSync | Made with ❤️</p>
</div>
"""
st.markdown(footer_css, unsafe_allow_html=True)

