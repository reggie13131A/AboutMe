import os
import re
import json
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Iterable, Optional, Set
from neo4j import AsyncGraphDatabase
import networkx as nx
import editdistance

# If you use langchain and neo4j integration:
try:
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_neo4j import Neo4jGraph
    from langchain_core.documents import Document
except Exception:
    # keep imports optional; user may adapt
    LLMGraphTransformer = None
    Neo4jGraph = None
    Document = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kg_dedup_pipeline")

def _safe_label(s: str) -> str:
    """Make a string safe to use as a Neo4j Label/RelType: keep A-Za-z0-9_ only, no leading digits."""
    if not s:
        return "Entity"
    s2 = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if re.match(r'^\d', s2):
        s2 = "_" + s2
    return s2 or "Entity"

async def write_graph_to_neo4j_async(G: nx.Graph, neo4j_config: dict,
                                     batch_size: int = 200,
                                     ensure_nodes_exist: bool = True):
    """
    Robust write: - explicitly use neo4j_config['database'] (default 'neo4j')
                  - verify node/edge counts after write
                  - optionally ensure nodes exist before creating edges
    """
    uri = neo4j_config["url"]
    user = neo4j_config["username"]
    pwd = neo4j_config["password"]
    database = neo4j_config.get("database", "neo4j")  # <-- explicit DB

    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))

    async with driver:
        async with driver.session(database=database) as session:
            try:
                await session.execute_write(lambda tx: tx.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.merged_id IS UNIQUE"))
            except Exception:
                pass

            nodes_by_label = defaultdict(list)
            expected_node_count = 0
            for nid, attrs in G.nodes(data=True):
                expected_node_count += 1
                label_raw = attrs.get("entity_type") or attrs.get("label") or "Entity"
                label = _safe_label(label_raw)
                props = dict(attrs.get("properties", {}) or {})

                aliases = list(attrs.get("aliases", []))
                if aliases:
                    props["aliases"] = aliases

                provenance = attrs.get("provenance", [])
                if provenance:
                    processed_provenance = []
                    for item in provenance:
                        if isinstance(item, dict):
                            # prefer 'content' if present
                            if "content" in item:
                                processed_provenance.append(item["content"])
                            else:
                                processed_provenance.append(json.dumps(item, ensure_ascii=False))
                        else:
                            processed_provenance.append(str(item))
                    props["provenance"] = processed_provenance

                props["confidence"] = float(attrs.get("confidence", 1.0))
                if attrs.get("pagerank") is not None:
                    try:
                        props["pagerank"] = float(attrs.get("pagerank"))
                    except Exception:
                        props["pagerank"] = str(attrs.get("pagerank"))
                props["label"] = attrs.get("label")
                props["merged_id"] = str(nid)

                # convert complex props -> primitives/arrays
                cleaned = {}
                for k, v in props.items():
                    if v is None:
                        cleaned[k] = None
                    elif isinstance(v, (str, int, float, bool)):
                        cleaned[k] = v
                    elif isinstance(v, (list, tuple)):
                        cleaned_list = []
                        for it in v:
                            if isinstance(it, (str, int, float, bool)):
                                cleaned_list.append(it)
                            else:
                                cleaned_list.append(json.dumps(it, ensure_ascii=False))
                        cleaned[k] = cleaned_list
                    else:
                        cleaned[k] = json.dumps(v, ensure_ascii=False)
                nodes_by_label[label].append({"merged_id": props["merged_id"], "props": cleaned})

            logger.info(f"准备写入节点数（期望）: {expected_node_count}")

            # write nodes grouped by label
            total_nodes_written = 0
            for label, items in nodes_by_label.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i:i+batch_size]
                    stmt = f"""
                    UNWIND $batch AS item
                    MERGE (n:{label} {{ merged_id: item.merged_id }})
                    SET n += item.props
                    """
                    await session.execute_write(lambda tx, q=stmt, b=batch: tx.run(q, batch=b))
                total_nodes_written += len(items)
            logger.info(f"节点写入完成（写入计数，按分组统计）: {total_nodes_written}")

            try:
                rec = await session.execute_read(lambda tx: tx.run("MATCH (n) RETURN count(n) AS cnt").single())
                db_node_count = rec["cnt"] if rec else None
                logger.info(f"数据库中节点总数: {db_node_count}")
                if db_node_count is not None and db_node_count < expected_node_count:
                    logger.warning("写入的节点比预期的少，后续会确保缺失节点以便写关系。")
                    if ensure_nodes_exist:
                        # build list of all merged_ids
                        all_nodes = [{"merged_id": str(nid)} for nid in G.nodes()]
                        for i in range(0, len(all_nodes), batch_size):
                            batch = all_nodes[i:i+batch_size]
                            stmt = """
                            UNWIND $batch AS item
                            MERGE (n { merged_id: item.merged_id })
                            """
                            await session.execute_write(lambda tx, q=stmt, b=batch: tx.run(q, batch=b))
                        logger.info("补全缺失的 minimal nodes 完成。")
            except Exception as e:
                logger.warning(f"节点计数/核验失败: {e}")

            edges_by_type = defaultdict(list)
            expected_edge_count = 0
            for u, v, ed in G.edges(data=True):
                expected_edge_count += 1
                rel_raw = None
                if isinstance(ed.get("types"), (list, set, tuple)) and len(ed.get("types")):
                    rel_raw = next(iter(ed.get("types")))
                rel_raw = rel_raw or ed.get("type") or "RELATED_TO"
                rel_type = _safe_label(rel_raw)
                props = dict(ed.get("properties", {}) or {})
                props["weight"] = ed.get("weight", props.get("weight", 1))
                props["confidence"] = float(ed.get("confidence", props.get("confidence", 1.0)))
                provenance = ed.get("provenance", [])
                if provenance:
                    proc = []
                    for item in provenance:
                        if isinstance(item, dict):
                            proc.append(item.get("content", json.dumps(item, ensure_ascii=False)))
                        else:
                            proc.append(str(item))
                    props["provenance"] = proc

                # sanitize props
                cleaned = {}
                for k, v in props.items():
                    if v is None:
                        cleaned[k] = None
                    elif isinstance(v, (str, int, float, bool)):
                        cleaned[k] = v
                    elif isinstance(v, (list, tuple)):
                        cleaned_list = []
                        for it in v:
                            if isinstance(it, (str, int, float, bool)):
                                cleaned_list.append(it)
                            else:
                                cleaned_list.append(json.dumps(it, ensure_ascii=False))
                        cleaned[k] = cleaned_list
                    else:
                        cleaned[k] = json.dumps(v, ensure_ascii=False)

                edges_by_type[rel_type].append({
                    "a_id": str(u),
                    "b_id": str(v),
                    "props": cleaned
                })

            logger.info(f"准备写入关系（期望）: {expected_edge_count}, 分组: {dict((k, len(v)) for k, v in edges_by_type.items())}")

            total_edges_written = 0
            for rel_type, items in edges_by_type.items():
                logger.info(f"写入关系类型 {rel_type} 共 {len(items)} 条")
                for i in range(0, len(items), batch_size):
                    batch = items[i:i+batch_size]
                    if ensure_nodes_exist:
                        stmt = f"""
                        UNWIND $batch AS e
                        MERGE (a {{ merged_id: e.a_id }})
                        MERGE (b {{ merged_id: e.b_id }})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += e.props
                        """
                    else:
                        stmt = f"""
                        UNWIND $batch AS e
                        MATCH (a {{ merged_id: e.a_id }}), (b {{ merged_id: e.b_id }})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += e.props
                        """
                    try:
                        await session.execute_write(lambda tx, q=stmt, b=batch: tx.run(q, batch=b))
                        total_edges_written += len(batch)
                    except Exception as e:
                        logger.exception(f"写入关系类型 {rel_type} 批次失败: {e}")
                        for bad in batch[:5]:
                            logger.error(f"失败条目: {bad}")
                        raise

            logger.info(f"关系写入完成（计数）: {total_edges_written}")

            try:
                rec = await session.execute_read(lambda tx: tx.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single())
                db_rel_count = rec["cnt"] if rec else None
                logger.info(f"数据库中关系总数: {db_rel_count}")
                # sample some relations for debugging
                sample = await session.execute_read(lambda tx: tx.run(
                    "MATCH (a)-[r]->(b) RETURN a.merged_id AS a_id, type(r) AS rel, b.merged_id AS b_id LIMIT 10").values())
                logger.info(f"关系样本 (a_id, rel, b_id): {sample}")
            except Exception as e:
                logger.warning(f"关系计数/采样失败: {e}")


# -----------------------------
# Utility dataclasses
# -----------------------------
@dataclass
class GraphChange:
    merged_groups: List[Dict[str, Any]]  # records of merges for auditing / rollback

    def __init__(self):
        self.merged_groups = []

    def add_merge(self, canonical, merged_nodes):
        self.merged_groups.append({"canonical": canonical, "merged": list(merged_nodes)})

# -----------------------------
# Similarity / blocking helpers
# -----------------------------
def _has_digit_in_2gram_diff(a: str, b: str) -> bool:
    def to_2gram_set(s: str):
        s = s or ""
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()

    set_a = to_2gram_set(a)
    set_b = to_2gram_set(b)
    diff = set_a ^ set_b
    return any(any(c.isdigit() for c in pair) for pair in diff)

def is_english_like(text: str) -> bool:
    if not text:
        return False
    # coarse english detector
    pattern = re.compile(r"[A-Za-z0-9\s\.,':;/\"?<>!\(\)\-]")
    # proportion of characters that match ascii-ish set
    matches = sum(1 for ch in text if pattern.match(ch))
    return (matches / max(1, len(text))) > 0.8

def is_similarity(a: str, b: str) -> bool:
    """
    Lightweight blocking similarity: adapted from ragflow code.
    Return True if likely similar (candidate for LLM disambiguation or immediate merge).
    """
    if not a or not b:
        return False
    if _has_digit_in_2gram_diff(a, b):
        return False

    if is_english_like(a) and is_english_like(b):
        # small edit distance
        return editdistance.eval(a, b) <= min(len(a), len(b)) // 2

    # for other langs, compare char overlap (rough)
    a_chars, b_chars = set(a), set(b)
    max_len = max(len(a_chars), len(b_chars))
    if max_len < 4:
        return len(a_chars & b_chars) > 1
    return (len(a_chars & b_chars) / max_len) >= 0.8


# -----------------------------
# Convert graph_documents <-> networkx
# -----------------------------
def graph_documents_to_nx(graph_documents: List[dict]) -> nx.Graph:
    """
        Convert a list of graph_documents (dict-like) into a NetworkX graph.
    """
    G = nx.Graph()
    for doc_idx, doc in enumerate(graph_documents):
        md = doc.get("metadata", {})
        for node in doc.get("nodes", []):
            nid = str(node["id"])
            attrs = {
                "label": node.get("label") or node.get("name") or nid,
                "entity_type": node.get("type") or node.get("entity_type") or "Entity",
                "properties": node.get("properties", {}),
                "aliases": set(node.get("properties", {}).get("aliases", []) or node.get("aliases", []) or []),
                "provenance": node.get("provenance", []) or [],  # list of evidence fragments
                "confidence": node.get("confidence", 1.0)
            }
            # If node already exists (same id across docs), merge attributes conservatively
            if G.has_node(nid):
                exist = G.nodes[nid]
                exist["aliases"].update(attrs["aliases"])
                exist["provenance"].extend(attrs["provenance"])
                # merge properties if missing
                for k, v in attrs["properties"].items():
                    if k not in exist["properties"] or not exist["properties"].get(k):
                        exist["properties"][k] = v
            else:
                G.add_node(nid, **attrs)

        for rel in doc.get("relationships", []):
            s = str(rel["source"])
            t = str(rel["target"])
            rtype = rel.get("type") or rel.get("relation") or "RELATED_TO"
            props = rel.get("properties", {}) or {}
            # accumulate a weight/evidence count to edge attributes
            ev_count = props.get("evidence_count", 1) if props else 1
            conf = props.get("confidence", 1.0)
            if G.has_edge(s, t):
                e = G[s][t]
                # aggregate weight and confidence
                e["weight"] = e.get("weight", 0) + ev_count
                e["confidence"] = max(e.get("confidence", 0), conf)
                # possibly keep list of relation types seen
                e["types"] = set(e.get("types", set())) | {rtype}
                e.setdefault("provenance", []).extend(rel.get("provenance", []))
            else:
                G.add_edge(s, t,
                           weight=ev_count,
                           confidence=conf,
                           types={rtype},
                           provenance=rel.get("provenance", []))
    return G

# 直接调用千问了
async def send_prompt_to_llm(llm, prompt_text: str, timeout: int = 240) -> str:
    """
    Robust wrapper: try several common LLM client patterns and normalize output to string.

    Priority:
      1. LangChain async generate: llm.agenerate([prompt_text])
      2. LangChain async generate_messages: llm.agenerate_messages([{"content":prompt_text}])
      3. LangChain async predict / apredict (if available)
      4. Sync methods in thread: llm.generate, llm.__call__, llm.chat, or model-specific SDKs
      5. Model-specific response shapes: .generations, .choices, .content, .text, dict with 'choices', etc.

    Returns empty string on failure (and logs the exception).
    """
    logger.debug("Sending prompt (len=%d) to llm", len(prompt_text))
    logger.debug("Prompt preview: %s", prompt_text[:400])

    # 1) Try common LangChain async generation
    try:
        if hasattr(llm, "agenerate"):
            logger.debug("Using llm.agenerate([...])")
            res = await llm.agenerate([prompt_text])
            # typical LangChain shape: res.generations[0][0].text
            try:
                return res.generations[0][0].text
            except Exception:
                # fallback: try to stringify
                return str(res)
        if hasattr(llm, "agenerate_messages"):
            logger.debug("Using llm.agenerate_messages([...])")
            # Some chat wrappers implement agenerate_messages
            res = await llm.agenerate_messages([{"content": prompt_text}])
            try:
                # try common shape
                return res.generations[0][0].message.content
            except Exception:
                return str(res)
    except Exception as e:
        logger.debug("Async LangChain-style methods failed: %s", e, exc_info=True)

    # 2) Try async predict-like methods (apredict/apredict_messages/predict)
    try:
        if hasattr(llm, "apredict"):
            logger.debug("Using llm.apredict(prompt_text)")
            out = await llm.apredict(prompt_text)
            return str(out)
        if hasattr(llm, "apredict_message") or hasattr(llm, "apredict_messages"):
            fn = getattr(llm, "apredict_messages", getattr(llm, "apredict_message"))
            logger.debug("Using llm.apredict_messages([...])")
            out = await fn([{"content": prompt_text}])
            return str(out)
        if hasattr(llm, "predict") and asyncio.iscoroutinefunction(llm.predict):
            logger.debug("Using async llm.predict")
            out = await llm.predict(prompt_text)
            return str(out)
    except Exception as e:
        logger.debug("apredict/predict-style methods failed: %s", e, exc_info=True)

    # 3) Fallback: call sync methods inside a thread pool
    def sync_call():
        try:
            # LangChain synchronous generate
            if hasattr(llm, "generate"):
                logger.debug("Trying sync llm.generate([...])")
                r = llm.generate([prompt_text])
                try:
                    return r.generations[0][0].text
                except Exception:
                    return str(r)
            # model.__call__(prompt)
            if hasattr(llm, "__call__"):
                logger.debug("Trying llm.__call__(prompt_text)")
                r = llm(prompt_text)
                return r if isinstance(r, str) else str(r)
            # some chat SDKs: chat / chat_completion
            if hasattr(llm, "chat"):
                logger.debug("Trying llm.chat(prompt_text)")
                r = llm.chat(prompt_text)
                # r may be object or dict
                if isinstance(r, dict):
                    # openai-like: choices[0].message.content
                    if r.get("choices"):
                        c = r["choices"][0]
                        if isinstance(c, dict) and c.get("message"):
                            return c["message"].get("content", str(r))
                        return c.get("text", str(r))
                    return str(r)
                if hasattr(r, "content"):
                    return r.content
                if hasattr(r, "text"):
                    return r.text
                return str(r)
            # fallback: some SDKs return dicts
            return ""
        except Exception as e:
            logger.exception("sync_call failed: %s", e)
            return ""

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, sync_call)
        return res or ""
    except RuntimeError:
        # no running loop (unlikely in async function) - call directly
        return sync_call()
    except Exception as e:
        logger.error("LLM call failed: %s", e, exc_info=True)
        return ""

# -----------------------------
# LLM-based resolution processing
# -----------------------------
def parse_llm_resolution_response(response_text: str,
                                  num_questions: int,
                                  record_delimiter: str = "##",
                                  entity_index_delimiter: str = "<|>",
                                  resolution_result_delimiter: str = "&&") -> List[int]:
    """
    Try to parse LLM response in two ways:
    1) JSON format: {"pairs":[{"i":1,"same":true}, ...]}
    2) fallback to Ragflow-style delimiters parsing: each record contains entity index.
    Returns list of indices (1-based) that are judged "yes" (same).
    """
    # 1) try JSON parse
    try:
        j = json.loads(response_text.strip())
        if isinstance(j, dict) and "pairs" in j:
            res = []
            for pair in j["pairs"]:
                if pair.get("same") or pair.get("same") is True:
                    idx = pair.get("i") or pair.get("idx") or pair.get("index")
                    if idx:
                        res.append(int(idx))
            return res
    except Exception:
        pass

    # 2) fallback to delimiter parsing similar to ragflow._process_results
    ans_list = []
    try:
        records = [r.strip() for r in response_text.split(record_delimiter)]
        for record in records:
            if not record:
                continue
            # find index
            pattern_int = f"{re.escape(entity_index_delimiter)}(\\d+){re.escape(entity_index_delimiter)}"
            match_int = re.search(pattern_int, record)
            res_int = int(str(match_int.group(1) if match_int else '0'))
            if res_int > num_questions:
                continue
            pattern_bool = f"{re.escape(resolution_result_delimiter)}([a-zA-Z]+){re.escape(resolution_result_delimiter)}"
            match_bool = re.search(pattern_bool, record)
            res_bool = str(match_bool.group(1) if match_bool else '')
            if res_int and res_bool and res_bool.lower().startswith("y"):
                ans_list.append(res_int)
    except Exception:
        pass

    return ans_list


async def resolve_candidates_with_llm(llm,
                                      candidate_pairs_by_type: Dict[str, List[Tuple[str, str]]],
                                      resolution_prompt_template: str,
                                      prompt_vars: dict = None,
                                      batch_size: int = 50,
                                      record_delimiter: str = "##",
                                      entity_index_delimiter: str = "<|>",
                                      resolution_result_delimiter: str = "&&") -> Set[Tuple[str, str]]:
    """
    candidate_pairs_by_type: { entity_type: [(a,b),(c,d), ...], ... }
    We batch each type's candidate pairs, format a prompt for each batch, call the LLM,
    parse the response, and return a set of (nodeA, nodeB) judged as same.
    Uses ragflow-style prompt formatting: the prompt template should accept replacements.
    """
    if prompt_vars is None:
        prompt_vars = {}

    resolved_pairs = set()

    # Helper to format one batch into a prompt text (mimic ragflow style)
    def build_batch_prompt(entity_type: str, pairs_batch: List[Tuple[str, str]]):
        lines = [f"When determining whether two {entity_type}s are the same, you should only focus on critical properties and overlook noisy factors.\n"]
        for idx, (a, b) in enumerate(pairs_batch, start=1):
            lines.append(f"Question {idx}: name of {entity_type} A is {a} ,name of {entity_type} B is {b}")
        sent = 'question above' if len(pairs_batch) == 1 else f'above {len(pairs_batch)} questions'
        lines.append(f"\nUse domain knowledge of {entity_type}s to help understand the text and answer the {sent} in the format: "
                     f"For Question i, Yes, {entity_type} A and {entity_type} B are the same {entity_type}./"
                     f"No, {entity_type} A and {entity_type} B are different {entity_type}s. "
                     f"For Question i+1, (repeat the above procedures)")
        pair_prompt = '\n'.join(lines)
        variables = {
            **(prompt_vars or {}),
            "input_text": pair_prompt,
            "record_delimiter": record_delimiter,
            "entity_index_delimiter": entity_index_delimiter,
            "resolution_result_delimiter": resolution_result_delimiter
        }
        # If the template uses Python format placeholders, use it
        try:
            prompt_text = resolution_prompt_template.format(**variables)
        except Exception:
            # fallback: insert raw pair_prompt
            prompt_text = pair_prompt
        return prompt_text

    # For each entity type, process batches
    for ent_type, pairs in candidate_pairs_by_type.items():
        if not pairs:
            continue
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            prompt_text = build_batch_prompt(ent_type, batch)
            logger.debug("LLM resolution prompt length=%d for %d pairs", len(prompt_text), len(batch))
            # send prompt to llm
            resp_text = await send_prompt_to_llm(llm, prompt_text)
            if not resp_text:
                logger.warning("Empty response from LLM for entity_type=%s batch start=%d", ent_type, i)
                continue
            # parse
            yes_indices = parse_llm_resolution_response(resp_text, len(batch),
                                                       record_delimiter, entity_index_delimiter,
                                                       resolution_result_delimiter)
            logger.debug("Parsed LLM yes indices: %s", yes_indices)
            for idx in yes_indices:
                try:
                    pair = batch[idx - 1]
                    a, b = pair
                    resolved_pairs.add((a, b))
                except Exception:
                    pass

    return resolved_pairs


# -----------------------------
# Merge components using PageRank + evidence + llm_confidence
# -----------------------------
def normalize_dict_values(d: Dict[Any, float]) -> Dict[Any, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def merge_components_using_pagerank(G: nx.Graph,
                                    resolution_pairs: Iterable[Tuple[str, str]],
                                    node_evidence_counts: Optional[Dict[str, int]] = None,
                                    pr_alpha: float = 0.85,
                                    weights: Tuple[float, float, float] = (0.6, 0.3, 0.1),
                                    graph_change: Optional[GraphChange] = None):
    """
    resolution_pairs: iterable of (a, b) node id strings judged identical.
    Steps:
      - build connected components graph from pairs
      - compute pagerank on G (original graph)
      - for each component pick canonical node by composite score:
            score = w1*norm(pr) + w2*norm(evidence_count) + w3*llm_confidence
      - merge other nodes into canonical (aggregate aliases, provenance, properties),
        redirect edges, merge duplicate edges (sum weights, max confidence)
      - recompute pagerank and write back to node attrs
    """
    if graph_change is None:
        graph_change = GraphChange()

    conn = nx.Graph()
    conn.add_edges_from(resolution_pairs)

    # compute pagerank on original graph (weighted if present)
    try:
        pr_scores = nx.pagerank(G, alpha=pr_alpha, weight='weight')
    except Exception as e:
        logger.warning("pagerank failed, falling back to degree-based approx: %s", e)
        pr_scores = {n: float(G.degree(n)) for n in G.nodes()}

    pr_norm = normalize_dict_values(pr_scores)
    evidence_norm = normalize_dict_values(node_evidence_counts or {})

    for comp in nx.connected_components(conn):
        if len(comp) == 1:
            continue
        # compute composite score
        composite = {}
        for n in comp:
            pr_s = pr_norm.get(n, 0.0)
            ev_s = evidence_norm.get(n, 0.0)
            llm_conf = float(G.nodes[n].get("llm_confidence", 0.0))
            composite[n] = weights[0] * pr_s + weights[1] * ev_s + weights[2] * llm_conf

        canonical = max(composite.keys(), key=lambda k: composite[k])
        merging_nodes = [n for n in comp if n != canonical]
        logger.info("Merging nodes %s into canonical %s (component size %d)", merging_nodes, canonical, len(comp))

        # aggregate attributes and redirect edges
        for m in merging_nodes:
            if not G.has_node(m) or not G.has_node(canonical):
                continue
            # attributes merge
            can_attrs = G.nodes[canonical]
            m_attrs = G.nodes[m]
            # aliases union
            can_aliases = set(can_attrs.get("aliases", set()))
            can_aliases.update(m_attrs.get("aliases", set()))
            can_attrs["aliases"] = can_aliases
            # provenance extend
            can_prov = can_attrs.get("provenance", [])
            can_prov.extend(m_attrs.get("provenance", []))
            can_attrs["provenance"] = can_prov
            # merge properties preferring canonical existing values
            for k, v in m_attrs.get("properties", {}).items():
                if k not in can_attrs.get("properties", {}) or not can_attrs["properties"].get(k):
                    can_attrs.setdefault("properties", {})[k] = v
            # merge confidence as max
            can_attrs["confidence"] = max(can_attrs.get("confidence", 0.0), m_attrs.get("confidence", 0.0))

            # redirect edges from m -> canonical, merging edge attrs
            for nbr in list(G.neighbors(m)):
                if nbr == canonical:
                    # skip self loop
                    continue
                data_m_n = dict(G.get_edge_data(m, nbr) or {})
                if G.has_edge(canonical, nbr):
                    # merge attributes
                    e = G[canonical][nbr]
                    e["weight"] = e.get("weight", 0) + data_m_n.get("weight", 0)
                    e["confidence"] = max(e.get("confidence", 0.0), data_m_n.get("confidence", 0.0))
                    e["types"] = set(e.get("types", set())) | set(data_m_n.get("types", []))
                    e.setdefault("provenance", []).extend(data_m_n.get("provenance", []))
                else:
                    G.add_edge(canonical, nbr,
                               weight=data_m_n.get("weight", 1),
                               confidence=data_m_n.get("confidence", 1.0),
                               types=set(data_m_n.get("types", [])),
                               provenance=list(data_m_n.get("provenance", [])))
            # finally remove node m
            G.remove_node(m)

        graph_change.add_merge(canonical, merging_nodes)

    # recompute pagerank after all merges
    try:
        new_pr = nx.pagerank(G, alpha=pr_alpha, weight='weight')
        for n, score in new_pr.items():
            if G.has_node(n):
                G.nodes[n]["pagerank"] = float(score)
    except Exception as e:
        logger.warning("recomputing pagerank failed: %s", e)

    return G, graph_change

def inspect_graph(G, top_k=10):
    print("总节点数:", G.number_of_nodes(), " 总边数:", G.number_of_edges())
    # 找没有 label 或 entity_type 的节点
    missing_label = [n for n,a in G.nodes(data=True) if not a.get("label")]
    print("没有 label 的节点数:", len(missing_label), "示例:", missing_label[:10])

    # 列出度数最高的几个节点（很可能是中心节点）
    degs = sorted(G.degree(), key=lambda x: -x[1])[:top_k]
    print("度数最高的节点 (id, degree):", degs)
    for n, d in degs:
        print("node:", n, "degree:", d, "attrs:", G.nodes[n])

# -----------------------------
# High-level pipeline
# -----------------------------
async def dedup_and_write_pipeline(texts: List[str],
                                   llm_for_extraction,
                                   llm_for_resolution,
                                   resolution_prompt: str,
                                   neo4j_config: dict,
                                   tmp_dir: Optional[str] = None):
    """
        texts: list of input documents (strings)
        llm_for_extraction: LLM wrapper used by LLMGraphTransformer (must be compatible)
        llm_for_resolution: LLM wrapper to call for candidate resolution (can be same as above)
        resolution_prompt: the prompt template string (Ragflow style). Our code will format it with {input_text} etc.
        neo4j_config: dict with keys url, username, password
    """
    # 0) Create documents for LangChain extraction
    if Document is None or LLMGraphTransformer is None:
        raise RuntimeError("LangChain graph transformer not available - please install/adjust imports.")

    docs = [Document(page_content=t) for t in texts]

    # 1) extract graph_documents via transformer
    logger.info("Running LLMGraphTransformer to extract graph_documents ...")
    # 这里明确要求大模型输出实体来源
    llm_transformer = LLMGraphTransformer(llm=llm_for_extraction)
    graph_documents = await llm_transformer.aconvert_to_graph_documents(docs)

    # For simplicity, convert to dict-of-lists form
    def lc_graphdoc_to_dict(gd):
        nodes = []
        # 获取 GraphDocument 的 source 信息
        graph_source = getattr(gd, "source", None)
        provenance_info = []
        if graph_source and hasattr(graph_source, 'page_content'):
            provenance_info = [{
                "content": graph_source.page_content,
                "metadata": getattr(graph_source, "metadata", {}),
            }]

        for n in gd.nodes:
            # handle if node is object or dict-like
            node_dict = {
                "id": getattr(n, "id", getattr(n, "node_id", None) or str(n)),
                "label": getattr(n, "label", None) or getattr(n, "name", None) or getattr(n, "id", None),
                "type": getattr(n, "type", None) or getattr(n, "entity_type", None) or None,
                "properties": getattr(n, "properties", {}) or {},
                "provenance": getattr(n, "provenance", []) or provenance_info,  # 使用 GraphDocument 的 source 信息
                "confidence": getattr(n, "confidence", 1.0)
            }
            nodes.append(node_dict)

        rels = []
        for r in gd.relationships:
            rels.append({
                "source": getattr(r.source, "id", getattr(r.source, "label", getattr(r.source, "node_id", str(r.source)))),
                "target": getattr(r.target, "id", getattr(r.target, "label", getattr(r.target, "node_id", str(r.target)))),
                "type": getattr(r, "type", getattr(r, "relation", None)) or "RELATED_TO",
                "properties": getattr(r, "properties", {}) or {},
                "provenance": getattr(r, "provenance", []) or []
            })
        # metadata是空的因为插件没下好，节点可以查看来源但是关系没有
        return {"nodes": nodes, "relationships": rels, "metadata": getattr(gd, "metadata", {}) or {}}

    # Normalize graph_documents into list of dicts (if needed)
    normalized_gds = []
    for gd in graph_documents:
        try:
            normalized_gds.append(lc_graphdoc_to_dict(gd))
        except Exception:
            # maybe already a dict
            normalized_gds.append(gd if isinstance(gd, dict) else {})

    logger.info("Extracted %d graph_documents", len(normalized_gds))

    # 2) Build a unified NetworkX graph
    G = graph_documents_to_nx(normalized_gds)
    print('原始的未经合并的graph：', G)

    # 3) Candidate generation: blocking by entity_type + is_similarity test
    logger.info("Generating candidate pairs with heuristic is_similarity...")
    nodes = list(G.nodes())
    nodes_by_type = defaultdict(list)
    for n, attrs in G.nodes(data=True):
        nodes_by_type[attrs.get("entity_type", "Entity")].append(n)
    # 准备要消除歧义的实体
    candidate_pairs_by_type = {}
    total_candidates = 0
    for ent_type, nodelist in nodes_by_type.items():
        pairs = []
        # naive O(N^2) within type; for large graphs replace with ANN / blocking
        for i in range(len(nodelist)):
            for j in range(i + 1, len(nodelist)):
                a = nodelist[i]; b = nodelist[j]
                la = G.nodes[a].get("label", "")
                lb = G.nodes[b].get("label", "")
                if is_similarity(la, lb):
                    pairs.append((a, b))
        candidate_pairs_by_type[ent_type] = pairs
        total_candidates += len(pairs)
    logger.info("Candidate pairs generated: %d", total_candidates)

    # 5) SECOND-ROUND: LLM resolution for candidate pairs
    logger.info("Resolving candidate pairs with LLM ...")
    resolved_pairs = await resolve_candidates_with_llm(llm_for_resolution,
                                                       candidate_pairs_by_type,
                                                       resolution_prompt_template=resolution_prompt,
                                                       prompt_vars=None,
                                                       batch_size=50)
    logger.info("LLM resolved %d pairs to merge", len(resolved_pairs))

    # 6) Merge components using PageRank + evidence counts (use edge weight as evidence proxy)
    node_evidence_counts = {n: len(G.nodes[n].get("provenance", [])) for n in G.nodes()}
    logger.info("Merging components based on LLM results and PageRank ...")
    G, change = merge_components_using_pagerank(G, resolved_pairs, node_evidence_counts=node_evidence_counts)
    inspect_graph(G, top_k=10)
    
    # write to Neo4j (langchain-neo4j)
    if neo4j_config:
        logger.info("Writing final graph (NetworkX) to Neo4j via direct writer ...")
        try:
            await write_graph_to_neo4j_async(G, neo4j_config, batch_size=200)
            logger.info("Direct Neo4j write succeeded.")
        except Exception as e:
            logger.exception("Direct Neo4j write failed: %s", e)
            # if desired, re-raise or return partial results
            raise
    else:
        logger.info("Neo4j write skipped (config missing).")

    logger.info("Pipeline complete. Merges recorded: %d groups", len(change.merged_groups))
    return normalized_gds, G, change


if __name__ == "__main__":
    import os
    from langchain_community.chat_models import ChatTongyi
    from ChatRobot.ideas.resolution_prompt import ENTITY_RESOLUTION_PROMPT
    async def run_example():
        os.environ["DASHSCOPE_API_KEY"] = "sk-7ac5a239fc414c79b36968ed1b6b9b0b"
        llm_for_extraction = ChatTongyi(
            model="qwen-max",
            temperature=0,
            max_tokens=4000
        )
        llm_for_resolution = llm_for_extraction
        resolution_prompt = ENTITY_RESOLUTION_PROMPT

        # Neo4j config
        neo4j_config = {
            "url": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
            "password": os.environ.get("NEO4J_PASSWORD", "Pensoon123")
        }

        texts = [
            "Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.",
            "Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize.",
            "Marie Sklodowska Curie was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice."
        ]

        final_gds, graph_obj, changes = await dedup_and_write_pipeline(texts,
                                                                       llm_for_extraction,
                                                                       llm_for_resolution,
                                                                       resolution_prompt,
                                                                       neo4j_config)
        logger.info("Final graph docs nodes: %d", len(final_gds[0]["nodes"]))
        logger.info("Merged groups: %d", len(changes.merged_groups))
        
        for i, group in enumerate(changes.merged_groups):
            logger.info(f"Group {i+1}: {group['canonical']} <- {group['merged']}")

    asyncio.run(run_example())