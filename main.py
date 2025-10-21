# backend/main.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import json
import re

app = FastAPI()

# Allow CORS for frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 GitHub token for authenticated requests
import os
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable not set. GPT features will error.")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

async def fetch_json(session: aiohttp.ClientSession, url: str, headers: Optional[Dict[str, str]] = None):
    async with session.get(url, headers=headers) as response:
        text = await response.text()
        if response.status >= 400:
            raise HTTPException(status_code=response.status, detail=f"HTTP {response.status} from {url}: {text[:300]}")
        return json.loads(text)

async def fetch_text(session: aiohttp.ClientSession, url: str):
    async with session.get(url) as response:
        if response.status != 200:
            return None
        return await response.text()

async def get_groq_response(messages: List[Dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        return "GROQ API key not configured (server)."
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            timeout=60
        ) as response:
            text = await response.text()
            if response.status != 200:
                return f"Error generating response: {response.status} - {text[:400]}"
            data = json.loads(text)
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(data)[:1000]

def parse_github_url(url: str) -> Optional[Dict[str, str]]:
    if not url:
        return None
    url = url.strip().replace("git@github.com:", "https://github.com/").replace("github.com:", "github.com/")
    m = re.search(r"(?:https?:\/\/)?(?:www\.)?github\.com\/(?P<owner>[^\/\s]+)\/(?P<repo>[^\/\s]+)", url)
    if not m:
        return None
    owner = m.group("owner")
    repo = m.group("repo").replace(".git", "")
    return {"owner": owner, "repo": repo}

def build_hierarchy(items: List[Dict[str, str]]) -> Dict[str, Any]:
    root = {"name": "/", "children": [], "path": ""}
    for item in items:
        parts = item["path"].split("/")
        node = root
        for i, p in enumerate(parts):
            child = next((c for c in node["children"] if c["name"] == p), None)
            if not child:
                child = {
                    "name": p,
                    "children": [],
                    "path": node["path"] + "/" + p if node["path"] else p,
                    "isFile": i == len(parts) - 1 and item.get("type") == "blob"
                }
                node["children"].append(child)
            node = child
    return root

def sort_tree(node: Dict[str, Any]):
    if "children" in node:
        node["children"].sort(key=lambda x: (not x.get("isFile", False), x["name"]))
        for child in node["children"]:
            sort_tree(child)

async def generate_summaries(node: Dict[str, Any], owner: str, repo: str, branch_sha: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore):
    if node.get("isFile"):
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch_sha}/{node['path']}"
        async with semaphore:
            txt = await fetch_text(session, raw_url)
        if txt:
            prompt = f"Provide a concise summary of this code file for software engineers. Highlight key functions, imports, and architecture. File: {node['path']}\n\n{txt}"
            node["summary"] = await get_groq_response([{"role": "user", "content": prompt}])
        else:
            node["summary"] = "Could not fetch file content."
    else:
        coros = [generate_summaries(c, owner, repo, branch_sha, session, semaphore) for c in node.get("children", [])]
        if coros:
            await asyncio.gather(*coros)

def collect_summaries(node: Dict[str, Any]) -> str:
    summaries = []
    def traverse(n):
        if n.get("isFile"):
            summaries.append(f"File: {n['path']}\nSummary: {n.get('summary', 'N/A')}\n")
        for c in n.get("children", []):
            traverse(c)
    traverse(node)
    return "\n".join(summaries)

@app.get("/api/repo")
async def get_repo(url: str):
    parsed = parse_github_url(url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")
    owner, repo = parsed["owner"], parsed["repo"]

    async with aiohttp.ClientSession() as session:
        try:
            repo_meta = await fetch_json(session, f"https://api.github.com/repos/{owner}/{repo}", headers=HEADERS)
            default_branch = repo_meta.get("default_branch", "main")

            ref = await fetch_json(session, f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{default_branch}", headers=HEADERS)
            commit_sha = ref["object"]["sha"]

            commit_obj = await fetch_json(session, f"https://api.github.com/repos/{owner}/{repo}/git/commits/{commit_sha}", headers=HEADERS)
            tree_sha = commit_obj["tree"]["sha"]

            tree_resp = await fetch_json(session, f"https://api.github.com/repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1", headers=HEADERS)
            blobs = [{"path": t["path"], "type": t["type"]} for t in tree_resp.get("tree", []) if t["type"] in ("blob", "tree")]

            root = build_hierarchy(blobs)
            semaphore = asyncio.Semaphore(6)
            await generate_summaries(root, owner, repo, tree_sha, session, semaphore)
            sort_tree(root)

            return {"tree": root, "summaries": collect_summaries(root), "repo": f"{owner}/{repo}", "fileCount": len(blobs)}
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_completion(request: ChatRequest):
    try:
        messages = [m.dict() for m in request.messages]
        response = await get_groq_response(messages)
        return {"content": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summary")
async def generate_summary(body: Dict[str, Any] = Body(...)):
    tree_str = json.dumps(body.get("tree", {}))
    summaries_str = body.get("summaries", "")
    prompt = f"Provide a professional overview of this repository. Analyze structure, tech stack, key components, and suggestions. Repo structure:\n{tree_str}\nSummaries:\n{summaries_str}"
    try:
        response = await get_groq_response([{"role": "user", "content": prompt}])
        return {"content": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
