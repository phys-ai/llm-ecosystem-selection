import argparse
import base64
import gzip
import json
import math
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SUBREDDITS: Tuple[str, ...] = (
    "changemyview",
    "explainlikeimfive",
    "technology",
    "science",
    "AskHistorians",
    "philosophy",
    "relationships",
    "TrueOffMyChest",
    "AmItheAsshole",
    "parenting",
    "AskReddit",
    "unpopularopinion",
    "news",
    "worldnews",
    "CasualConversation",
    "selfimprovement",
)

SUBREDDIT_GROUPS: Dict[str, str] = {
    "changemyview": "analytic_reasoning",
    "explainlikeimfive": "analytic_reasoning",
    "technology": "analytic_reasoning",
    "science": "analytic_reasoning",
    "AskHistorians": "public_reasoning",
    "philosophy": "identity_reasoning",
    "relationships": "interpersonal_conflict",
    "TrueOffMyChest": "emotional_disclosure",
    "AmItheAsshole": "moral_judgment",
    "parenting": "family_dynamics",
    "AskReddit": "engagement_dynamics",
    "unpopularopinion": "engagement_dynamics",
    "news": "public_reasoning",
    "worldnews": "public_reasoning",
    "CasualConversation": "online_behavior",
    "selfimprovement": "everyday_growth",
    "careerguidance": "everyday_growth",
    "personalfinance": "everyday_growth",
    "TooAfraidToAsk": "identity_reasoning",
    "mentalhealth": "emotional_disclosure",
}

PRONOUN_PATTERNS = [
    (r"\bmy friend\b", "a friend"),
    (r"\bmy partner\b", "a partner"),
    (r"\bmy parents\b", "parents"),
    (r"\bmy\b", "a"),
    (r"\bme\b", "someone"),
    (r"\bi\b", "someone"),
    (r"\bwe\b", "people"),
    (r"\bour\b", "people's"),
    (r"\bus\b", "people"),
    (r"\bhe\b", "a person"),
    (r"\bshe\b", "a person"),
    (r"\bthey\b", "a person"),
    (r"\bhim\b", "that person"),
    (r"\bher\b", "that person"),
    (r"\bthem\b", "that person"),
]

EMOTION_WORDS = {
    "feel", "hurt", "angry", "sad", "anxious", "upset", "guilty", "jealous", "lonely",
    "embarrassed", "mad", "cry", "resent", "ashamed", "offended", "frustrated",
}
INTERPERSONAL_WORDS = {
    "friend", "partner", "boyfriend", "girlfriend", "wife", "husband", "family", "parent",
    "mom", "dad", "relationship", "coworker", "roommate", "teacher", "child", "kids",
}
EVIDENCE_WORDS = {
    "evidence", "proof", "study", "research", "data", "claim", "source", "reason", "logic",
    "science", "statistics", "credible", "verified", "history", "facts",
}
ENGAGEMENT_WORDS = {
    "viral", "attention", "backlash", "controversial", "unpopular", "reply", "comments",
    "debate", "support", "dragged", "ratio", "hot take", "thread",
}
SAFETY_WORDS = {
    "suicide", "self-harm", "murder", "kill", "abuse", "assault", "overdose", "violence",
    "nsfw", "porn", "rape",
}

FILTERED_FLAIR_TERMS = {"meme", "shitpost", "image", "gif", "video"}
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")
AGE_GENDER_RE = re.compile(r"\[\s*\d{1,2}\s*[mfnbMFNB/]+\s*\]|\(\s*\d{1,2}\s*[mfnbMFNB/]+\s*\)")
BRACKET_RE = re.compile(r"\[[^\]]+\]|\([^\)]*\)")
TITLE_PREFIX_RE = re.compile(
    r"^(aita|am i the asshole|tifu|cmv|eli5|serious|discussion|question|advice needed|help)\s*[:\-]?\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawCandidate:
    source: str
    source_subreddit: str
    source_post_id: str
    title: str
    selftext_excerpt: str
    top_comments_excerpt: Tuple[str, ...]
    raw_topic_text: str
    permalink: str
    score: int
    num_comments: int
    created_utc: float
    over_18: bool = False


@dataclass(frozen=True)
class TopicBankEntry:
    text: str
    source: str
    source_subreddit: str
    source_post_id: str
    normalized_from: str
    title: str
    selftext_excerpt: str
    top_comments_excerpt: Tuple[str, ...]
    raw_topic_text: str
    controversy: float
    emotional_load: float
    interpersonalness: float
    evidentiality: float
    engagement_baitness: float
    public_vs_personal: float
    safety_risk_hint: float
    coarse_category: str


class RedditOAuthClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        timeout_sec: float = 30.0,
        sleep_between_calls_sec: float = 0.2,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.sleep_between_calls_sec = sleep_between_calls_sec
        self._token: str = ""
        self._token_expires_at: float = 0.0
        self._ssl_context = build_ssl_context()

    def _request(self, req: urllib.request.Request) -> Any:
        max_retries = 6
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec, context=self._ssl_context) as resp:
                    body = resp.read().decode("utf-8")
                    if self.sleep_between_calls_sec > 0:
                        time.sleep(self.sleep_between_calls_sec)
                    return json.loads(body)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        wait_sec = float(retry_after) if retry_after else min(15.0 * (attempt + 1), 120.0)
                    except ValueError:
                        wait_sec = min(15.0 * (attempt + 1), 120.0)
                    print(f"[rate-limit] sleeping {wait_sec:.1f}s after 429 from {req.full_url}", flush=True)
                    time.sleep(wait_sec)
                    continue
                if 500 <= exc.code < 600 and attempt < max_retries:
                    wait_sec = min(5.0 * (attempt + 1), 30.0)
                    print(f"[server-error] sleeping {wait_sec:.1f}s after HTTP {exc.code} from {req.full_url}", flush=True)
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"HTTP {exc.code} for {req.full_url}: {body[:500]}") from exc

    def _ensure_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expires_at - 30:
            return self._token

        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("ascii")
        req = urllib.request.Request(
            "https://www.reddit.com/api/v1/access_token",
            data=data,
            headers={
                "Authorization": f"Basic {basic}",
                "User-Agent": self.user_agent,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        payload = self._request(req)
        self._token = str(payload["access_token"])
        self._token_expires_at = now + float(payload.get("expires_in", 3600))
        return self._token

    def get_listing(self, subreddit: str, sort: str, limit: int, after: str = "", timeframe: str = "year") -> Dict[str, Any]:
        token = self._ensure_token()
        params = {"limit": str(limit), "raw_json": "1"}
        if after:
            params["after"] = after
        if sort == "top":
            params["t"] = timeframe
        qs = urllib.parse.urlencode(params)
        url = f"https://oauth.reddit.com/r/{subreddit}/{sort}.json?{qs}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": self.user_agent},
        )
        payload = self._request(req)
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected listing payload for r/{subreddit}: {type(payload).__name__}")
        return payload

    def get_top_comments(self, post_id: str, limit: int = 3) -> Tuple[str, ...]:
        token = self._ensure_token()
        params = urllib.parse.urlencode({"limit": str(limit), "sort": "top", "depth": "1", "raw_json": "1"})
        url = f"https://oauth.reddit.com/comments/{post_id}.json?{params}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "User-Agent": self.user_agent},
        )
        payload = self._request(req)
        comments: List[str] = []
        if isinstance(payload, list) and len(payload) >= 2:
            listing = payload[1]
            children = (((listing or {}).get("data") or {}).get("children") or [])
            for child in children:
                data = (child or {}).get("data") or {}
                body = clean_text(str(data.get("body", "")))
                if body and body not in {"[removed]", "[deleted]"}:
                    comments.append(body[:280])
                if len(comments) >= limit:
                    break
        return tuple(comments)


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text or "")
    text = text.replace("&amp;", "&").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def build_ssl_context() -> ssl.SSLContext:
    cafile_candidates = [
        os.environ.get("SSL_CERT_FILE", ""),
        "/etc/ssl/cert.pem",
        "/private/etc/ssl/cert.pem",
    ]
    for cafile in cafile_candidates:
        if cafile and os.path.exists(cafile):
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def subreddit_prior_axes(subreddit: str) -> Dict[str, float]:
    analytic = {"changemyview", "explainlikeimfive", "technology", "science", "AskHistorians", "philosophy"}
    emotion = {"relationships", "TrueOffMyChest", "AmItheAsshole", "parenting", "mentalhealth"}
    engagement = {"AskReddit", "unpopularopinion", "news", "worldnews"}
    baseline = {"CasualConversation", "selfimprovement", "careerguidance", "personalfinance", "TooAfraidToAsk"}
    axes = {
        "controversy": 0.45,
        "emotional_load": 0.35,
        "interpersonalness": 0.30,
        "evidentiality": 0.40,
        "engagement_baitness": 0.35,
        "public_vs_personal": 0.50,
    }
    if subreddit in analytic:
        axes.update({"evidentiality": 0.80, "interpersonalness": 0.18, "emotional_load": 0.18, "public_vs_personal": 0.82})
    elif subreddit in emotion:
        axes.update({"emotional_load": 0.80, "interpersonalness": 0.88, "evidentiality": 0.25, "public_vs_personal": 0.16})
    elif subreddit in engagement:
        axes.update({"engagement_baitness": 0.72, "controversy": 0.62, "public_vs_personal": 0.76})
    elif subreddit in baseline:
        axes.update({"public_vs_personal": 0.45, "emotional_load": 0.36, "interpersonalness": 0.42, "evidentiality": 0.45})
    return axes


def estimate_axes(candidate: RawCandidate, normalized_prompt: str) -> Dict[str, float]:
    tokens = set(tokenize(" ".join([candidate.title, candidate.selftext_excerpt, " ".join(candidate.top_comments_excerpt), normalized_prompt])))
    priors = subreddit_prior_axes(candidate.source_subreddit)
    emotion_hits = sum(1 for token in tokens if token in EMOTION_WORDS)
    interpersonal_hits = sum(1 for token in tokens if token in INTERPERSONAL_WORDS)
    evidence_hits = sum(1 for token in tokens if token in EVIDENCE_WORDS)
    engagement_hits = sum(1 for token in tokens if token in ENGAGEMENT_WORDS)
    safety_hits = sum(1 for token in tokens if token in SAFETY_WORDS)
    question_bonus = 1 if "?" in candidate.title or "?" in normalized_prompt else 0
    score_norm = math.log1p(max(candidate.score, 0)) / 10.0
    comment_norm = math.log1p(max(candidate.num_comments, 0)) / 8.0

    controversy = clamp01(priors["controversy"] + 0.05 * engagement_hits + 0.04 * question_bonus + 0.06 * comment_norm)
    emotional_load = clamp01(priors["emotional_load"] + 0.08 * emotion_hits + 0.03 * interpersonal_hits)
    interpersonalness = clamp01(priors["interpersonalness"] + 0.09 * interpersonal_hits)
    evidentiality = clamp01(priors["evidentiality"] + 0.08 * evidence_hits - 0.04 * emotion_hits)
    engagement_baitness = clamp01(priors["engagement_baitness"] + 0.08 * engagement_hits + 0.05 * question_bonus + 0.05 * score_norm)
    public_vs_personal = clamp01(priors["public_vs_personal"] + 0.07 * evidence_hits - 0.07 * interpersonal_hits)
    safety_risk_hint = clamp01(0.05 + 0.18 * safety_hits + (0.15 if candidate.over_18 else 0.0))
    return {
        "controversy": controversy,
        "emotional_load": emotional_load,
        "interpersonalness": interpersonalness,
        "evidentiality": evidentiality,
        "engagement_baitness": engagement_baitness,
        "public_vs_personal": public_vs_personal,
        "safety_risk_hint": safety_risk_hint,
    }


def is_probably_bad_candidate(post: Dict[str, Any]) -> bool:
    title = clean_text(str(post.get("title", "")))
    selftext = clean_text(str(post.get("selftext", "")))
    flair = clean_text(str(post.get("link_flair_text", ""))).lower()
    if not title or len(title) < 16:
        return True
    if post.get("stickied") or post.get("pinned"):
        return True
    if str(post.get("author", "")) in {"[deleted]", "AutoModerator"}:
        return True
    if bool(post.get("over_18")):
        return True
    if flair and flair in FILTERED_FLAIR_TERMS:
        return True
    if post.get("is_video"):
        return True
    if str(post.get("post_hint", "")) in {"image", "hosted:video", "rich:video"}:
        return True
    lowered = f"{title} {selftext}".lower()
    if any(term in lowered for term in SAFETY_WORDS):
        return True
    return False


def normalize_title_to_prompt(title: str, subreddit: str) -> str:
    text = clean_text(title)
    text = AGE_GENDER_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = TITLE_PREFIX_RE.sub("", text)
    text = clean_text(text).rstrip(".!")
    if not text:
        return ""

    lowered = text.lower()
    if " or " in lowered and not lowered.endswith("?"):
        return capitalize_first(text) + "?"
    if lowered.startswith("why "):
        return capitalize_first(text if text.endswith("?") else text + "?")
    if lowered.startswith("how "):
        return capitalize_first(text if text.endswith("?") else text + "?")
    if lowered.startswith("when "):
        return capitalize_first(text if text.endswith("?") else text + "?")
    if lowered.startswith("what "):
        return capitalize_first(text if text.endswith("?") else text + "?")
    if lowered.startswith("should "):
        return capitalize_first(text if text.endswith("?") else text + "?")
    if lowered.startswith("is it ") or lowered.startswith("is ") or lowered.startswith("are "):
        return capitalize_first(text if text.endswith("?") else text + "?")
    if lowered.startswith("am i the asshole"):
        return ""
    if subreddit.lower() in {"amitheasshole", "relationships", "parenting", "trueoffmychest"}:
        converted = text
        for pattern, replacement in PRONOUN_PATTERNS:
            converted = re.sub(pattern, replacement, converted, flags=re.IGNORECASE)
        converted = clean_text(converted)
        converted = converted.replace("someone stop inviting that person", "they are no longer invited")
        converted = converted.replace("someone stop inviting a person", "they are no longer invited")
        converted = converted.replace("someone stop", "someone stops")
        converted = converted.replace("a stop", "someone stops")
        converted = converted.replace("gets upset when they are no longer invited", "still expects inclusion after repeated cancellations")
        if not converted:
            return ""
        return capitalize_first(f"How should people respond when {converted.lower()}?")
    return capitalize_first(f"What does it reveal when {text.lower()}?")


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def openai_normalize_candidate(candidate: RawCandidate, model: str, api_key: str, timeout_sec: float) -> Optional[str]:
    prompt = f"""
You are normalizing a Reddit post into a reusable, safe discussion prompt.

Return JSON only:
{{
  "normalized_prompt": string
}}

Requirements:
- 1 or 2 sentences
- no proper names or subreddit-specific jargon
- preserve the core issue
- should invite opinion, feeling, or analysis
- avoid explicit unsafe framing
- present a discussion question, not an answer

Subreddit: {candidate.source_subreddit}
Title: {candidate.title}
Selftext excerpt: {candidate.selftext_excerpt}
Top comments: {json.dumps(candidate.top_comments_excerpt, ensure_ascii=False)}
""".strip()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You normalize social discussion prompts. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=build_ssl_context()) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        prompt_text = clean_text(str(parsed.get("normalized_prompt", "")))
        return prompt_text or None
    except Exception:
        return None


def build_topic_entry(candidate: RawCandidate, use_llm: bool, llm_model: str, timeout_sec: float) -> Optional[TopicBankEntry]:
    normalized_prompt = ""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if use_llm and api_key:
        normalized_prompt = openai_normalize_candidate(candidate, llm_model, api_key, timeout_sec) or ""
    if not normalized_prompt:
        normalized_prompt = normalize_title_to_prompt(candidate.title, candidate.source_subreddit)
    normalized_prompt = clean_text(normalized_prompt)
    if len(normalized_prompt) < 20:
        return None

    axes = estimate_axes(candidate, normalized_prompt)
    if axes["safety_risk_hint"] >= 0.7:
        return None

    return TopicBankEntry(
        text=normalized_prompt,
        source=candidate.source,
        source_subreddit=candidate.source_subreddit,
        source_post_id=candidate.source_post_id,
        normalized_from=candidate.title,
        title=candidate.title,
        selftext_excerpt=candidate.selftext_excerpt,
        top_comments_excerpt=candidate.top_comments_excerpt,
        raw_topic_text=candidate.raw_topic_text,
        controversy=axes["controversy"],
        emotional_load=axes["emotional_load"],
        interpersonalness=axes["interpersonalness"],
        evidentiality=axes["evidentiality"],
        engagement_baitness=axes["engagement_baitness"],
        public_vs_personal=axes["public_vs_personal"],
        safety_risk_hint=axes["safety_risk_hint"],
        coarse_category=SUBREDDIT_GROUPS.get(candidate.source_subreddit, candidate.source_subreddit.lower()),
    )


def dedupe_entries(entries: Sequence[TopicBankEntry]) -> List[TopicBankEntry]:
    seen_exact: set[str] = set()
    seen_signatures: List[Tuple[set[str], TopicBankEntry]] = []
    kept: List[TopicBankEntry] = []
    for entry in entries:
        exact_key = entry.text.lower()
        if exact_key in seen_exact:
            continue
        token_set = set(tokenize(entry.text))
        if len(token_set) < 4:
            continue
        duplicate = False
        for prev_tokens, _ in seen_signatures:
            overlap = len(token_set & prev_tokens) / max(1, len(token_set | prev_tokens))
            if overlap >= 0.82:
                duplicate = True
                break
        if duplicate:
            continue
        seen_exact.add(exact_key)
        seen_signatures.append((token_set, entry))
        kept.append(entry)
    return kept


def vectorize_entry(entry: TopicBankEntry) -> Tuple[float, ...]:
    return (
        entry.controversy,
        entry.emotional_load,
        entry.interpersonalness,
        entry.evidentiality,
        entry.engagement_baitness,
        entry.public_vs_personal,
        entry.safety_risk_hint,
    )


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def sample_diverse_entries(entries: Sequence[TopicBankEntry], sample_size: int, rng: random.Random) -> List[TopicBankEntry]:
    if sample_size <= 0 or sample_size >= len(entries):
        return list(entries)
    by_subreddit: Dict[str, List[TopicBankEntry]] = {}
    for entry in entries:
        by_subreddit.setdefault(entry.source_subreddit, []).append(entry)
    selected: List[TopicBankEntry] = []
    for subreddit in sorted(by_subreddit):
        selected.append(rng.choice(by_subreddit[subreddit]))
        if len(selected) >= sample_size:
            return selected[:sample_size]

    remaining = [entry for entry in entries if entry not in selected]
    while len(selected) < sample_size and remaining:
        if not selected:
            choice = rng.choice(remaining)
        else:
            choice = max(
                remaining,
                key=lambda entry: min(euclidean_distance(vectorize_entry(entry), vectorize_entry(prev)) for prev in selected),
            )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def iter_dump_posts(path: str) -> Iterable[Dict[str, Any]]:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == "[":
            payload = json.load(f)
            for row in payload:
                if isinstance(row, dict):
                    yield row
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def candidate_from_post(post: Dict[str, Any], source_subreddit: str, top_comments: Sequence[str], source_name: str) -> Optional[RawCandidate]:
    if is_probably_bad_candidate(post):
        return None
    title = clean_text(str(post.get("title", "")))
    selftext = clean_text(str(post.get("selftext", "")))[:500]
    raw_topic = clean_text(" ".join(part for part in [title, selftext] if part))
    permalink = str(post.get("permalink", ""))
    if permalink and permalink.startswith("/"):
        permalink = f"https://www.reddit.com{permalink}"
    candidate = RawCandidate(
        source=source_name,
        source_subreddit=source_subreddit,
        source_post_id=str(post.get("id", "")),
        title=title,
        selftext_excerpt=selftext,
        top_comments_excerpt=tuple(clean_text(comment)[:280] for comment in top_comments if clean_text(comment)),
        raw_topic_text=raw_topic[:700],
        permalink=permalink,
        score=int(post.get("score", 0) or 0),
        num_comments=int(post.get("num_comments", 0) or 0),
        created_utc=float(post.get("created_utc", 0.0) or 0.0),
        over_18=bool(post.get("over_18", False)),
    )
    return candidate


def collect_from_dump_paths(paths: Sequence[str], subreddits: Sequence[str], posts_per_subreddit: int) -> List[RawCandidate]:
    allowed = {sub.lower() for sub in subreddits}
    counters: Dict[str, int] = {}
    candidates: List[RawCandidate] = []
    for path in paths:
        for post in iter_dump_posts(path):
            subreddit = str(post.get("subreddit", ""))
            if subreddit.lower() not in allowed:
                continue
            if counters.get(subreddit.lower(), 0) >= posts_per_subreddit:
                continue
            candidate = candidate_from_post(post, subreddit, (), "reddit_dump")
            if candidate is None:
                continue
            candidates.append(candidate)
            counters[subreddit.lower()] = counters.get(subreddit.lower(), 0) + 1
    return candidates


def collect_from_reddit_api(
    subreddits: Sequence[str],
    posts_per_subreddit: int,
    sorts: Sequence[str],
    timeframe: str,
    user_agent: str,
    timeout_sec: float,
    sleep_between_calls_sec: float,
) -> List[RawCandidate]:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set for Reddit API collection.")

    client = RedditOAuthClient(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        timeout_sec=timeout_sec,
        sleep_between_calls_sec=sleep_between_calls_sec,
    )
    all_candidates: List[RawCandidate] = []
    per_subreddit_cap = max(1, posts_per_subreddit)
    for subreddit in subreddits:
        print(f"[collect] r/{subreddit}: target={per_subreddit_cap}", flush=True)
        collected: List[RawCandidate] = []
        seen_ids: set[str] = set()
        for sort in sorts:
            after = ""
            while len(collected) < per_subreddit_cap:
                batch_size = min(100, per_subreddit_cap - len(collected))
                payload = client.get_listing(subreddit, sort=sort, limit=batch_size, after=after, timeframe=timeframe)
                children = (((payload or {}).get("data") or {}).get("children") or [])
                if not children:
                    break
                progress_made = False
                for child in children:
                    data = (child or {}).get("data") or {}
                    post_id = str(data.get("id", ""))
                    if not post_id or post_id in seen_ids:
                        continue
                    try:
                        comments = client.get_top_comments(post_id, limit=3)
                    except Exception as exc:
                        print(f"[warn] r/{subreddit} post={post_id}: comment fetch failed, continuing without comments ({exc})", flush=True)
                        comments = ()
                    candidate = candidate_from_post(data, subreddit, comments, "reddit_api")
                    seen_ids.add(post_id)
                    if candidate is None:
                        continue
                    collected.append(candidate)
                    progress_made = True
                    if len(collected) % 25 == 0 or len(collected) == per_subreddit_cap:
                        print(f"[collect] r/{subreddit}: {len(collected)}/{per_subreddit_cap}", flush=True)
                    if len(collected) >= per_subreddit_cap:
                        break
                after = str((((payload or {}).get("data") or {}).get("after")) or "")
                if not after or not progress_made:
                    break
        print(f"[collect] r/{subreddit}: done={len(collected)}", flush=True)
        all_candidates.extend(collected[:per_subreddit_cap])
    return all_candidates


def write_jsonl(rows: Sequence[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_topic_bank(
    raw_candidates: Sequence[RawCandidate],
    use_llm: bool,
    llm_model: str,
    timeout_sec: float,
    sample_size: int,
    random_seed: int,
) -> Tuple[List[TopicBankEntry], List[TopicBankEntry]]:
    normalized: List[TopicBankEntry] = []
    for candidate in raw_candidates:
        entry = build_topic_entry(candidate, use_llm=use_llm, llm_model=llm_model, timeout_sec=timeout_sec)
        if entry is not None:
            normalized.append(entry)
    bank = dedupe_entries(normalized)
    bank.sort(key=lambda row: (row.source_subreddit.lower(), -row.controversy, -row.engagement_baitness, row.text.lower()))
    sampled = sample_diverse_entries(bank, sample_size=sample_size, rng=random.Random(random_seed))
    sampled.sort(key=lambda row: (row.source_subreddit.lower(), row.text.lower()))
    return bank, sampled


def parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["reddit_api", "dump", "both"], default="reddit_api")
    parser.add_argument("--subreddits", default=",".join(DEFAULT_SUBREDDITS))
    parser.add_argument("--posts_per_subreddit", type=int, default=300)
    parser.add_argument("--sorts", default="top,new")
    parser.add_argument("--timeframe", default="year")
    parser.add_argument("--dump_paths", nargs="*", default=[])
    parser.add_argument("--output", default="topic_bank.jsonl")
    parser.add_argument("--sampled_output", default="topic_bank.sampled.jsonl")
    parser.add_argument("--raw_output", default="topic_candidates.raw.jsonl")
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--user_agent", default="script:evolution_theory.topic_bank:0.1 (by /u/yourusername)")
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    parser.add_argument("--sleep_between_calls_sec", type=float, default=0.2)
    parser.add_argument("--normalize_with_llm", action="store_true")
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    args = parser.parse_args()

    subreddits = parse_csv(args.subreddits)
    sorts = parse_csv(args.sorts)
    if not subreddits:
        raise ValueError("subreddits must be non-empty")
    if not sorts:
        raise ValueError("sorts must be non-empty")

    raw_candidates: List[RawCandidate] = []
    if args.source in {"reddit_api", "both"}:
        raw_candidates.extend(
            collect_from_reddit_api(
                subreddits=subreddits,
                posts_per_subreddit=args.posts_per_subreddit,
                sorts=sorts,
                timeframe=args.timeframe,
                user_agent=args.user_agent,
                timeout_sec=args.timeout_sec,
                sleep_between_calls_sec=args.sleep_between_calls_sec,
            )
        )
    if args.source in {"dump", "both"}:
        if not args.dump_paths:
            raise ValueError("dump source requires at least one --dump_paths entry")
        raw_candidates.extend(collect_from_dump_paths(args.dump_paths, subreddits=subreddits, posts_per_subreddit=args.posts_per_subreddit))

    if not raw_candidates:
        write_jsonl([], args.raw_output)
        write_jsonl([], args.output)
        write_jsonl([], args.sampled_output)
        print("Collected raw candidates: 0")
        print(f"Wrote full topic bank: {args.output} (0 rows)")
        print(f"Wrote sampled topic bank: {args.sampled_output} (0 rows)")
        print(f"Wrote raw candidate archive: {args.raw_output}")
        return

    bank, sampled = build_topic_bank(
        raw_candidates=raw_candidates,
        use_llm=args.normalize_with_llm,
        llm_model=args.llm_model,
        timeout_sec=args.timeout_sec,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )
    write_jsonl([asdict(row) for row in raw_candidates], args.raw_output)
    write_jsonl([asdict(row) for row in bank], args.output)
    write_jsonl([asdict(row) for row in sampled], args.sampled_output)

    print(f"Collected raw candidates: {len(raw_candidates)}")
    print(f"Wrote full topic bank: {args.output} ({len(bank)} rows)")
    print(f"Wrote sampled topic bank: {args.sampled_output} ({len(sampled)} rows)")
    print(f"Wrote raw candidate archive: {args.raw_output}")


if __name__ == "__main__":
    main()
