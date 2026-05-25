import json
from typing import List, Dict, Any

from openai import OpenAI

from rag import RAGPipeline

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the indexed document knowledge base for chunks relevant to the query. "
                "Always call this before answering factual questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default 5, max 10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List all documents currently indexed in the knowledge base.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = (
    "You are SmartDoc Assistant. Your only knowledge source is the chunks returned by search_documents. "
    "Rules you must never break:\n"
    "1. Always call search_documents before answering any factual question.\n"
    "2. Only use information that appears verbatim or is directly inferable from the retrieved chunks. "
    "Do NOT supplement with your own training knowledge.\n"
    "3. If the retrieved chunks do not contain enough information to answer, say exactly: "
    "'The uploaded documents do not contain enough information to answer this question.' "
    "Do not guess, infer beyond the text, or fabricate sources.\n"
    "4. When you do answer, cite the source filename for every claim."
)


class SmartDocAgent:
    def __init__(self, rag: RAGPipeline, api_key: str):
        self.rag = rag
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"

    def _execute_tool(self, name: str, args: Dict) -> str:
        if name == "search_documents":
            results = self.rag.search(args["query"], n_results=args.get("n_results", 5))
            if not results:
                return "No relevant content found in the knowledge base."
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(
                    f"[Chunk {i}] Source: {r['source']} | Relevance: {r['score']}\n{r['content']}"
                )
            return "\n\n---\n\n".join(parts)

        if name == "list_documents":
            docs = self.rag.list_documents()
            if not docs:
                return "No documents are indexed yet."
            return "Indexed documents:\n" + "\n".join(f"  • {d}" for d in docs)

        return f"Unknown tool: {name}"

    def chat(self, history: List[Dict[str, str]], user_message: str) -> Dict[str, Any]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # Agentic tool-calling loop
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                new_history = list(history) + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": msg.content},
                ]
                return {"answer": msg.content, "history": new_history}

            # Append assistant message with tool calls
            messages.append(msg)

            # Execute every tool call and feed results back
            for tc in msg.tool_calls:
                result = self._execute_tool(
                    tc.function.name, json.loads(tc.function.arguments)
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

    def stream_chat(self, history: List[Dict[str, str]], user_message: str):
        """
        Generator for streaming responses.

        Phase 1 (non-streaming): let DeepSeek call tools and collect results.
        Phase 2 (streaming):     synthesise the final answer token-by-token.

        Yields dicts:
          {"type": "token",  "content": "<text>"}
          {"type": "done",   "history": [...]}
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # Phase 1 — tool-calling (non-streaming, DeepSeek may call multiple tools in parallel)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Edge case: DeepSeek skipped tools entirely — return as single chunk
            content = msg.content or ""
            yield {"type": "token", "content": content}
            yield {"type": "done", "history": list(history) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
            ]}
            return

        messages.append(msg)
        for tc in msg.tool_calls:
            result = self._execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # Phase 2 — streaming synthesis
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )

        full_content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_content += token
                yield {"type": "token", "content": token}

        yield {"type": "done", "history": list(history) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_content},
        ]}
