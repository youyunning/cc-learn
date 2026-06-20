import ast, json, os, subprocess, time, re
from pathlib import Path
import yaml
try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

WORKDIR = Path.cwd()
client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
)
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
MEMORY_DIR = WORKDIR / ".memory"; MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

# s06: subagent gets its own system prompt — no task, no recursion
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)

# ═══════════════════════════════════════════════════════════
#  NEW in s09: Memory System
# ═══════════════════════════════════════════════════════════
MEMORY_TYPE = ["user", "feedback", "project", "reference"]

def list_memory_files():
    """List all memory files with metadata."""
    result = []
    for f in  sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return result
def select_relevant_memories(messages: list,max_item: int = 5) ->list[str]:
     """Select relevant memory filenames by matching recent conversation against
    memory names/descriptions. Uses a simple LLM call (or falls back to keyword
    matching on name+description)."""
     files = list_memory_files()
     if not files:
         return []
     #提取用户最近的对话注入上下文
     recent_texts = []
     for msg in reversed(messages):
         if msg.get("role") == "user":
             content = msg.get(" content","")
             if isinstance(content,list):
                 content = " ".join(
                     str(getattr(b, "text", "")) for b in content
                     if getattr(b, "type", None) == "text"
                 )
             if isinstance(content,str):
                 recent_texts.append(content)
             if len(recent_texts) >= 3:
                 break
     recent = " ".join(reversed(recent_texts))[:2000]

     if not recent.strip():
         return []
     # 建立 name + description 的索引给LLM 去选择
     catalog_lines = []
     for i, f in enumerate(files):
         catalog_lines.append(f"-{i}: {f['name']} -= {f['description']}\n")
     catalog = "\n".join(catalog_lines)

     prompt = (
         "Given the recent conversation and the memory catalog below, "
         "select the indices of memories that are clearly relevant. "
         "Return ONLY a JSON array of integers, e.g. [0, 3]. "
         "If none are relevant, return [].\n\n"
         f"Recent conversation:\n{recent}\n\n"
         f"Memory catalog:\n{catalog}"
     )

     try:
         response = client.messages.create(
             model=MODEL,
             messages=[{"role": "user", "content": prompt}],
             max_tokens=200,
         )
         text = extract_text(response.content).strip()
         # 提取 json数组从大模型返回的
         match = re.search(r'\[.*?\]', text, re.DOTALL)
         if match:
             indices = json.loads(match.group())
             selectd = []
             for idx in indices:
                 if isinstance(idx,int) and 0 <= idx < len(files):
                     selectd.append(files[idx]["filename"])
                     if len(selectd) >= max_item:
                         break
             return  selectd
     except Exception as e:
         pass


def load_memories(messages: list) -> str:
    """Load relevant memory content for injection into context."""
    select_files = select_relevant_memories(messages)
    if not select_files:
        return ""
    parts = ["<relevant_memories>"]
    for f in select_files:
        content = read_memory_file(filename=f)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)





def write_memory_file(name:str, mem_type:str, description: str, body: str):
    """Write a single memory file with YAML frontmatter."""
    slug = name.lower().replace(" ","-").replace("/","-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
    )
    _rebuild_index()
    return filepath

def _rebuild_index():
    """Rebuild MEMORY.md index from all memory files."""
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")
def read_memory_index() ->str:
    """Read MEMORY.md index (injected into SYSTEM every turn)."""
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""

def read_memory_file(filename: str) -> str | None:
    """Read a single memory file's full content."""
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()

def extract_memories(messages: list):
    """Extract new memories from recent dialogue. Runs after each turn."""
    dialogue_parts = []
    for msg in messages[-10:]:  # only look at the last 10 messages
        role = msg.get("role","?")
        content = msg.get("content","")
        if isinstance(content, list):
            content = "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content.strip()}")
    dialogue = "\n".join(dialogue_parts)
    if not dialogue.strip():
        return

    # 检查存在的记忆去避免重复
    existing = list_memory_files()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )
    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=800
        )
        text = extract_text(response.content).strip()
        # Extract JSON array from response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type","user")
            desc = mem.get("description","")
            body = mem.get("body","")
            if desc and body:
                write_memory_file()
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass

CONSOLIDATE_THRESHOLD = 10

def consolidate_memories():
    """Merge duplicate/stale memories. Triggered when file count ≥ threshold."""
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000
        )
        text = extract_text(response.content).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)

        print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
    except Exception:
        pass




def build_system() -> str:
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"You are a coding agent at {WORKDIR}."
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )

# ═══════════════════════════════════════════════════════════
#  NEW in s08: Four-Layer Compaction Pipeline
# ═══════════════════════════════════════════════════════════

CONTEXT_LIMIT = 50000
KEEP_RECENT = 3
PERSIST_THRESHOLD = 30000

def estimate_size(msgs): return len(str(msgs))

def _block_type(block):
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _message_has_tool_use(msg):
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg):
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result"
               for block in content)


# L1: snipCompact — trim middle messages
def snip_compact(messages, max_messages=50):
    if len(messages) <= max_messages: return messages
    keep_head, keep_tail = 3, max_messages - 3
    head_end, tail_start = keep_head, len(messages) - keep_tail
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    return messages[:head_end] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[tail_start:]


# L2: microCompact — old result placeholders
def collect_tool_results(messages):
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks

def micro_compact(messages):
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT: return messages
    for _, _, block in tool_results[:-KEEP_RECENT]:
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


# L3: toolResultBudget — persist large results to disk
def persist_large_output(tool_use_id, output):
    if len(output) <= PERSIST_THRESHOLD: return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists(): path.write_text(output)
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

def tool_result_budget(messages, max_bytes=200_000):
    last = messages[-1] if messages else None
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return messages
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes: return messages
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)
    for _, block in ranked:
        if total <= max_bytes: break
        content = str(block.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD: continue
        tid = block.get("tool_use_id", "unknown")
        block["content"] = persist_large_output(tid, content)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


# L4: autoCompact — LLM full summary
def write_transcript(messages):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages: f.write(json.dumps(msg, default=str) + "\n")
    return path

def summarize_history(messages):
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text").strip() or "(empty summary)"

def compact_history(messages):
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# Emergency: reactiveCompact — on API error
def reactive_compact(messages):
    transcript = write_transcript(messages)
    summary = summarize_history(messages)
    tail_start = max(0, len(messages) - 5)
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]

# ═══════════════════════════════════════════════════════════
#  NEW in s07: SKILL技能
# ═══════════════════════════════════════════════════════════
SKILL_REGISTRY: dict[str, dict] = {}

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()
def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}

_scan_skills()

def list_skills() -> str:
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())
def build_system_prompt() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

SYSTEM = build_system_prompt()

# --工具调用 函数，执行bash命令并返回结果
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Command is too dangerous to run."
    try:
        result = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        return (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

# 新增4工具调用
def safe_path(path: str) -> Path:
    path = (WORKDIR / path).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return path
def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"
def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern,root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "No matches found."
    except Exception as e:
        return f"Error: {e}"
def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

def extract_text(content) -> str:
    """Extract text from message content blocks."""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
def spawn_subagent(description: str) -> str:
    """Spawn a subagent with fresh messages[], return summary only."""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]  # fresh context

    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # Issue 1: subagent also runs hooks (permissions apply)
                blocked = trigger_hook("PreToolUse", block)
                if blocked:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                trigger_hook("PostToolUse", block, output)
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})
        # Issue 5: fallback if safety limit hit during tool_use

    result = extract_text(messages[-1]["content"])
    if not result:
        # last message is tool_result, look backwards for assistant text
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result  # only summary, entire message history discarded
# ═══════════════════════════════════════════════════════════
#  NEW in s02: 工具定义（s01 只有一个 bash，现在扩展到 5 个）
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
{"name": "todo_write", "description": "Create and manage a task list ...",
     "input_schema": {
         "type": "object",
         "properties": {
             "todos": {
                 "type": "array",
                 "items": {
                     "type": "object",
                     "properties": {
                         "content": {"type": "string"},
                         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                     },
                 },
             },
         },
     },
    },
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
{
    "name": "task",
    "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
},
]

# ═══════════════════════════════════════════════════════════
#  NEW in s02: 工具分发映射（s01 是硬编码 run_bash，现在改为查表）
# ═══════════════════════════════════════════════════════════

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,  "todo_write": run_todo_write,
    "task": spawn_subagent, "load_skill": load_skill,
}
rounds_since_todo = 0

MAX_REACTIVE_RETRIES = 1
def agent_loop(messages: list):
    global rounds_since_todo
    reactive_retries = 0
    # s09: inject relevant memory content into the current user turn
    memories_content = load_memories(messages)
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None
    # s09: build system once per user turn; memory is updated after the loop returns
    system = build_system()

    while True:
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role", ""),
                                                       "content": str(m.get("content", ""))} for m in messages]
        # s08 change: three preprocessors (0 API calls, cheap first)
        # Order matches CC source: budget → snip → micro
        messages[:] = tool_result_budget(messages)    # L3: persist large results first
        messages[:] = snip_compact(messages)          # L1: trim middle
        messages[:] = micro_compact(messages)         # L2: old result placeholders

        # s08 change: tokens still over threshold → LLM summary (1 API call)
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        # 注入todo_write工具的当前任务列表
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        try:
            request_messages = messages
            if memories_content and memory_turn is not None and memory_turn < len(messages):
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
                }
            request_messages = messages
            if memories_content and memory_turn is not None and memory_turn < len(messages):
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
                }
            response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
            )
            reactive_retries = 0  # reset on successful API call
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(
                    e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            raise
        # 将大模型的回复加入messages中
        messages.append({"role": "assistant", "content": response.content})
        # 如果大模型不使用工具就做完任务
        if response.stop_reason != "tool_use":
            force = trigger_hook("Stop", messages)
            # s09: extract from pre-compression snapshot for full fidelity
            extract_memories(pre_compress)
            consolidate_memories()
            if force:
                messages.append({"role": "user", "content": force})
                continue  # 继续循环，不退出
            return
        rounds_since_todo += 1
        # 执行工具调用，收集结果
        results = []
        for block in response.content:
            if block.type != "tool_use": continue
            print(f"\033[36m> {block.name}\033[0m")

            # s08: compact tool triggers compact_history, not a no-op string
            if block.name == "compact":
                # Pop the assistant message we just added so compact_history
                # only compresses older turns. Then re-add it so the
                # tool_use → tool_result pair stays intact for the API.
                last_assistant = messages.pop()
                messages[:] = compact_history(messages)
                messages.append(last_assistant)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                messages.append({"role": "user", "content": results})
                break  # end current turn, start fresh with compacted context


            blocked = trigger_hook("PreToolUse", block)
            if blocked is not None:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"No handler for tool {block.name}"
            trigger_hook("PostToolUse", block, output)

            if block.name == "todo_write":
                rounds_since_todo = 0

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        # 将assistant的工具调用和工具结果接入messages中，继续对话
        messages.append({"role": "user", "content": results})

# ═══════════════════════════════════════════════════════════
#  NEW in s03: 权限配置
# ═══════════════════════════════════════════════════════════
#规则一：禁止执行危险命令
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot", "> /dev/", "passwd", "chown", "chmod", "mkfs", "dd", "kill", "pkill",
]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]
def check_deny_list(command: str) -> str | None:
    for d in DENY_LIST:
        if d in command:
            return f"Command is too dangerous to run: {d}"
    return  None
#规则二： 规则匹配
PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "File path escapes workspace.",
    }
]
def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

# 规则三：命中后，暂停等用户输入
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"

# ═══════════════════════════════════════════════════════════
#  NEW in s04: Hook注入
# ═══════════════════════════════════════════════════════════
#定义Hook生命周期
HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}
# 注册Hook
def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hook(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result # 返回值 不等于 None hook就返回
    return None
# 用户输入提交后、进入LLM前触发
def context_inject_hook(query: str) -> str | None:
    """Inject current working directory info into every prompt."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None   # return None = no modification, let prompt through
def permission_hook(block) :
    if block.name == "bash":
        for d in DENY_LIST:
            if d in block.input.get("command", ""):
                print(f"\n⛔ [HOOK]Command is too dangerous to run: {d}")
                return "Permission denied by user"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(f"\n\033[33m[HOOK] Potentially destructive command\033[0m")
                print(f"   [HOOK]Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.name in ["write_file", "edit_file"]:
        if not (WORKDIR / block.input.get("path", "")).resolve().is_relative_to(WORKDIR):
            choice = input(f"\n⚠ File path escapes workspace: {block.input.get('path')}. Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

# PreToolUse hook: before tool use, check permission
def log_hook(block):
    print(f"\033[90m[HOOK] PreToolUse: {block.name}({block.input})\033[0m")
# PostToolUse: 大文件提醒
def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[HOOK] ⚠ Large output from {block.name}")
    else:
        print(f"\033[90m[HOOK] PostToolUse: {block.name} output length {len(str(output))}\033[0m")
#循环退出时候触发
def summary_hook(messages: list) -> str | None:
    """Print a summary when the loop is about to stop."""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None   # return None = allow stop, return string = force continuation

register_hook("Stop", summary_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("UserPromptSubmit", context_inject_hook)
# ═══════════════════════════════════════════════════════════
#  NEW in s05: tode_write
# ═══════════════════════════════════════════════════════════
CURRENT_TODOS: list[dict] = []

def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None

# ═══════════════════════════════════════════════════════════
#  NEW in s06: 子agent
# ═══════════════════════════════════════════════════════════

SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]
# NO "task" tool — prevent recursive spawning

SUB_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}








if __name__ == "__main__":
    print("输入问题，回车发送。输入q退出。\n")
    # 用来记录与大模型的历史对话
    history =  []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hook("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # 打印与模型对话的最后一行的回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()

