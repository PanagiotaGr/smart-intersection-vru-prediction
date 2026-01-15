#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import feedparser
import requests
import yaml
from dateutil import parser as dtparser

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Python <3.9 not supported in Actions anyway


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    published_utc: str
    updated_utc: str
    primary_category: str
    categories: List[str]
    pdf_url: str
    abs_url: str


# ----------------------------
# Helpers
# ----------------------------

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip().lower()


def today_in_tz(tz_name: str) -> datetime:
    if ZoneInfo is None:
        # Fallback: UTC
        return datetime.utcnow()
    return datetime.now(ZoneInfo(tz_name))


def to_tz_date(dt_utc: datetime, tz_name: str) -> datetime.date:
    if ZoneInfo is None:
        return dt_utc.date()
    return dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo(tz_name)).date()


# ----------------------------
# arXiv fetching
# ----------------------------

def build_arxiv_url(endpoint: str, search_query: str, max_results: int) -> str:
    # We request submittedDate sorting for stable ordering
    # arXiv API uses Atom feed.
    q = quote(search_query, safe="():\"' ORAND+-_*")
    return (
        f"{endpoint}"
        f"?search_query={q}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )


def parse_arxiv_entry(entry: Any) -> Paper:
    # entry.id is like http://arxiv.org/abs/XXXX.YYYYYvN
    abs_url = entry.get("id", "").strip()
    arxiv_id = abs_url.rsplit("/", 1)[-1]

    title = re.sub(r"\s+", " ", entry.get("title", "").strip())
    summary = re.sub(r"\s+", " ", entry.get("summary", "").strip())

    authors = []
    for a in entry.get("authors", []) or []:
        name = a.get("name", "").strip()
        if name:
            authors.append(name)

    published = entry.get("published", "").strip()
    updated = entry.get("updated", "").strip()

    # categories / primary category
    tags = entry.get("tags", []) or []
    categories = [t.get("term", "").strip() for t in tags if t.get("term")]
    primary_category = entry.get("arxiv_primary_category", {}).get("term", "").strip()
    if not primary_category and categories:
        primary_category = categories[0]

    # pdf url
    pdf_url = ""
    for link in entry.get("links", []) or []:
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href", "").strip()
            break
    if not pdf_url:
        pdf_url = abs_url.replace("/abs/", "/pdf/")

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        summary=summary,
        published_utc=published,
        updated_utc=updated,
        primary_category=primary_category,
        categories=categories,
        pdf_url=pdf_url,
        abs_url=abs_url,
    )


def fetch_arxiv(endpoint: str, search_query: str, max_results: int, timeout_s: int = 30) -> List[Paper]:
    url = build_arxiv_url(endpoint, search_query, max_results)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    papers: List[Paper] = []
    for entry in feed.entries:
        try:
            papers.append(parse_arxiv_entry(entry))
        except Exception:
            # Skip malformed entry (rare) to avoid breaking the run
            continue
    return papers


# ----------------------------
# Filtering & selection
# ----------------------------

def keyword_filter(
    paper: Paper,
    include_keywords: List[str],
    exclude_keywords: List[str],
) -> bool:
    text = normalize_text(paper.title + " " + paper.summary)

    # Exclude wins immediately
    for kw in exclude_keywords:
        if normalize_text(kw) in text:
            return False

    # If include list is empty => accept
    if not include_keywords:
        return True

    # Require at least one include keyword hit
    for kw in include_keywords:
        if normalize_text(kw) in text:
            return True
    return False


def category_filter(paper: Paper, allowed_categories: List[str]) -> bool:
    if not allowed_categories:
        return True
    cats = set(paper.categories + ([paper.primary_category] if paper.primary_category else []))
    return any(c in cats for c in allowed_categories)


def sort_papers_deterministic(papers: List[Paper]) -> List[Paper]:
    def key(p: Paper):
        # published desc, then title asc, then id asc
        try:
            pub = dtparser.isoparse(p.published_utc)
        except Exception:
            pub = datetime(1970, 1, 1)
        return (-int(pub.timestamp()), p.title.lower(), p.arxiv_id)

    return sorted(papers, key=key)


# ----------------------------
# Persistence
# ----------------------------

def paper_to_dict(p: Paper) -> Dict[str, Any]:
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "authors": p.authors,
        "summary": p.summary,
        "published_utc": p.published_utc,
        "updated_utc": p.updated_utc,
        "primary_category": p.primary_category,
        "categories": p.categories,
        "pdf_url": p.pdf_url,
        "abs_url": p.abs_url,
    }


def dict_to_paper(d: Dict[str, Any]) -> Paper:
    return Paper(
        arxiv_id=d["arxiv_id"],
        title=d["title"],
        authors=list(d.get("authors", [])),
        summary=d.get("summary", ""),
        published_utc=d.get("published_utc", ""),
        updated_utc=d.get("updated_utc", ""),
        primary_category=d.get("primary_category", ""),
        categories=list(d.get("categories", [])),
        pdf_url=d.get("pdf_url", ""),
        abs_url=d.get("abs_url", ""),
    )


# ----------------------------
# Markdown rendering
# ----------------------------

def render_paper_md(p: Paper) -> str:
    # Compact, thesis-friendly
    # One bullet with links + metadata.
    try:
        pub = dtparser.isoparse(p.published_utc).date().isoformat()
    except Exception:
        pub = "unknown-date"

    authors = ", ".join(p.authors[:8])
    if len(p.authors) > 8:
        authors += " et al."

    title = p.title.replace("\n", " ").strip()

    return (
        f"- **{title}**  \n"
        f"  *{authors}*  \n"
        f"  Published: `{pub}` · Category: `{p.primary_category or 'n/a'}` · "
        f"[abs]({p.abs_url}) · [pdf]({p.pdf_url}) · id: `{p.arxiv_id}`\n"
    )


def render_topic_page(topic_name: str, topic_slug: str, papers: List[Paper], tz_name: str) -> str:
    now = today_in_tz(tz_name)
    header = (
        f"# {topic_name}\n\n"
        f"Updated: `{now.date().isoformat()}` (timezone: `{tz_name}`)\n\n"
        f"Total papers tracked: **{len(papers)}**\n\n"
        f"---\n\n"
    )
    body = "".join(render_paper_md(p) for p in papers)
    return header + body


def render_digest(digest_date: str, topic_sections: List[Tuple[str, List[Paper]]], tz_name: str) -> str:
    now = today_in_tz(tz_name)
    lines = [
        f"# Daily ArXiv Digest — {digest_date}\n",
        f"Generated: `{now.isoformat(timespec='seconds')}` (timezone: `{tz_name}`)\n",
        "\n---\n\n",
    ]
    total_today = sum(len(ps) for _, ps in topic_sections)
    lines.append(f"Total new papers today (across topics): **{total_today}**\n\n")

    for topic_name, papers in topic_sections:
        lines.append(f"## {topic_name} ({len(papers)})\n\n")
        if not papers:
            lines.append("_No new papers matched today._\n\n")
            continue
        for p in papers:
            lines.append(render_paper_md(p))
        lines.append("\n")

    return "".join(lines)


# ----------------------------
# README update (between markers)
# ----------------------------

README_MARKER_START = "<!-- TOPICS:START -->"
README_MARKER_END = "<!-- TOPICS:END -->"
LATEST_MARKER_START = "<!-- LATEST:START -->"
LATEST_MARKER_END = "<!-- LATEST:END -->"


def update_readme(
    readme_path: Path,
    topics_info: List[Dict[str, Any]],
    latest_date: str,
    latest_digest_relpath: str,
) -> None:
    if not readme_path.exists():
        return

    text = readme_path.read_text(encoding="utf-8")

    # Latest block
    latest_block = (
        f"{LATEST_MARKER_START}\n"
        f"- Updated on: **{latest_date}**\n"
        f"- Latest digest: `{latest_digest_relpath}`\n"
        f"{LATEST_MARKER_END}"
    )
    text = replace_block(text, LATEST_MARKER_START, LATEST_MARKER_END, latest_block)

    # Topics table block
    table_lines = []
    table_lines.append("| Topic | Latest Update | Papers | Link |")
    table_lines.append("|------|--------------:|------:|------|")
    for t in topics_info:
        table_lines.append(
            f"| {t['name']} | {t['latest_update']} | {t['count']} | {t['link']} |"
        )
    topics_block = (
        f"{README_MARKER_START}\n"
        + "\n".join(table_lines)
        + f"\n{README_MARKER_END}"
    )
    text = replace_block(text, README_MARKER_START, README_MARKER_END, topics_block)

    readme_path.write_text(text, encoding="utf-8")


def replace_block(full_text: str, start: str, end: str, new_block: str) -> str:
    if start in full_text and end in full_text:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
        return pattern.sub(new_block, full_text, count=1)
    # If markers missing, append at end
    return full_text.rstrip() + "\n\n" + new_block + "\n"


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yml")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    tz_name = cfg.get("project", {}).get("timezone", "UTC")

    storage = cfg.get("storage", {})
    data_dir = Path(storage.get("data_dir", "data"))
    db_file = data_dir / storage.get("db_file", "papers.json")

    digests_dir = Path(storage.get("digests_dir", "digests"))
    topics_dir = Path(storage.get("topics_dir", "topics"))

    data_dir.mkdir(parents=True, exist_ok=True)
    digests_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)

    endpoint = cfg.get("arxiv", {}).get("endpoint", "http://export.arxiv.org/api/query")
    max_results = int(cfg.get("arxiv", {}).get("max_results_per_topic", 80))
    allowed_categories = cfg.get("arxiv", {}).get("allowed_categories", []) or []

    global_inc = cfg.get("filters", {}).get("include_keywords", []) or []
    global_exc = cfg.get("filters", {}).get("exclude_keywords", []) or []

    topics = cfg.get("topics", []) or []
    if not topics:
        raise SystemExit("No topics defined in config.yml")

    # Load DB
    db = read_json(db_file, default={"papers": {}, "topics": {}})

    # Compute "today" in desired timezone
    now_tz = today_in_tz(tz_name)
    digest_date = now_tz.date().isoformat()

    # We'll collect "today's" matched papers per topic
    topic_sections: List[Tuple[str, List[Paper]]] = []
    topics_info: List[Dict[str, Any]] = []

    max_daily_per_topic = int(cfg.get("output", {}).get("max_daily_per_topic", 12))

    for t in topics:
        name = t["name"]
        slug = t.get("slug") or slugify(name)
        query = t["query"]

        t_inc = (t.get("include_keywords") or [])
        t_exc = (t.get("exclude_keywords") or [])

        # Fetch from arXiv
        fetched = fetch_arxiv(endpoint, query, max_results=max_results)

        # Filter by category + keywords
        filtered: List[Paper] = []
        for p in fetched:
            if not category_filter(p, allowed_categories):
                continue
            if not keyword_filter(p, global_inc + t_inc, global_exc + t_exc):
                continue
            filtered.append(p)

        filtered = sort_papers_deterministic(filtered)

        # Insert into DB (dedupe by arxiv_id)
        topic_key = slug
        if "topics" not in db:
            db["topics"] = {}
        if topic_key not in db["topics"]:
            db["topics"][topic_key] = {"name": name, "papers": [], "latest_update": "1970-01-01"}

        seen_ids = set(db["topics"][topic_key]["papers"])
        new_ids_for_topic: List[str] = []

        for p in filtered:
            if p.arxiv_id not in db["papers"]:
                db["papers"][p.arxiv_id] = paper_to_dict(p)

            if p.arxiv_id not in seen_ids:
                new_ids_for_topic.append(p.arxiv_id)
                seen_ids.add(p.arxiv_id)

        # Update per-topic list
        if new_ids_for_topic:
            db["topics"][topic_key]["papers"].extend(new_ids_for_topic)

        # Rebuild per-topic papers list (Paper objects) and sort deterministically
        topic_papers: List[Paper] = []
        for pid in db["topics"][topic_key]["papers"]:
            try:
                topic_papers.append(dict_to_paper(db["papers"][pid]))
            except Exception:
                continue
        topic_papers = sort_papers_deterministic(topic_papers)

        # Update latest_update for topic
        latest_update = db["topics"][topic_key].get("latest_update", "1970-01-01")
        if digest_date > latest_update and (new_ids_for_topic or latest_update == "1970-01-01"):
            # Even if no new today, keep latest_update as last time we actually ran successfully today:
            db["topics"][topic_key]["latest_update"] = digest_date
            latest_update = digest_date

        # Write topic page
        topic_md = render_topic_page(name, slug, topic_papers, tz_name)
        topic_path = topics_dir / f"{slug}.md"
        topic_path.write_text(topic_md, encoding="utf-8")

        # Select today papers for digest: match published date == today (in tz)
        todays: List[Paper] = []
        for p in filtered:
            try:
                pub_utc = dtparser.isoparse(p.published_utc)
            except Exception:
                continue
            if to_tz_date(pub_utc, tz_name).isoformat() == digest_date:
                todays.append(p)

        todays = sort_papers_deterministic(todays)[:max_daily_per_topic]
        topic_sections.append((name, todays))

        # Topic navigator info
        topics_info.append(
            {
                "name": name,
                "latest_update": latest_update,
                "count": len(topic_papers),
                "link": f"[{name}](topics/{slug}.md)",
            }
        )

    # Write digest
    digest_path = digests_dir / f"{digest_date}.md"
    digest_md = render_digest(digest_date, topic_sections, tz_name)
    digest_path.write_text(digest_md, encoding="utf-8")

    # Persist DB
    write_json(db_file, db)

    # Update README
    update_readme(
        readme_path=Path("README.md"),
        topics_info=topics_info,
        latest_date=digest_date,
        latest_digest_relpath=f"digests/{digest_date}.md",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
