import requests
from typing import Dict, Any, Optional


class LangFlowClient:
    """Client for LangFlow API"""

    def __init__(self, langflow_url: str, flow_id: str, api_key: Optional[str] = None):
        if not flow_id:
            raise ValueError("LANGFLOW_FLOW_ID must be provided")

        self.langflow_url = langflow_url.rstrip("/")
        self.flow_id = flow_id
        self.api_key = api_key

    def trigger_summarization(self, paper_id: str) -> Dict[str, Any]:
        """
        Trigger LangFlow summarization flow and extract Notion URL
        """
        try:
            headers = {
                "Content-Type": "application/json"
            }

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "input_value": paper_id,
                "input_type": "chat"
            }

            response = requests.post(
                f"{self.langflow_url}/api/v1/run/{self.flow_id}",
                json=payload,
                headers=headers,
                timeout=120
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"LangFlow error {response.status_code}: {response.text}"
                }

            result = response.json()

            # 1️⃣ Prefer Notion URL from Notion node output
            notion_url = self._extract_notion_url_from_result(result)

            # 2️⃣ Fallback: extract from ChatOutput text (robust)
            if not notion_url:
                output_text = self._extract_output_text(result)
                notion_url = self._extract_notion_url_from_text(output_text)

            return {
                "success": True,
                "paper_id": paper_id,
                "notion_url": notion_url,
                "raw_result": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


    def _extract_notion_url_from_result(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract Notion page URL directly from Notion Create Page node output
        """
        try:
            for output in result.get("outputs", []):
                for item in output.get("outputs", []):
                    results = item.get("results", {})
                    # Notion node usually returns a page object
                    page = results.get("page") or results.get("result") or {}
                    if isinstance(page, dict):
                        url = page.get("url")
                        if url and "notion.so" in url:
                            return url
        except Exception:
            pass
        return None

    def _extract_output_text(self, result: Dict[str, Any]) -> str:
        """
        Safely extract ChatOutput text without fragile indexing
        """
        try:
            for output in result.get("outputs", []):
                for item in output.get("outputs", []):
                    message = item.get("results", {}).get("message", {})
                    if isinstance(message, dict) and "text" in message:
                        return message["text"]
        except Exception:
            pass
        return ""

    def _extract_notion_url_from_text(self, text: str) -> Optional[str]:
        """
        Extract Notion URL from plain text (fallback only)
        """
        import re

        match = re.search(r"https://www\.notion\.so/[^\s]+", text)
        return match.group(0) if match else None