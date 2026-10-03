import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TraitProfile:
    position: str
    specialization: str
    objective_orientation: str


@dataclass(frozen=True)
class ContinuousTraitProfile:
    coords: Dict[str, float]
    labels: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class Agent:
    agent_id: str
    trait: Any
    post_model: Optional["ModelSpec"] = None


@dataclass(frozen=True)
class Evaluator:
    evaluator_id: str
    trait: Any
    eval_model: Optional["ModelSpec"] = None


@dataclass(frozen=True)
class Topic:
    text: str
    source: str = "synthetic"
    source_subreddit: str = ""
    source_post_id: str = ""
    normalized_from: str = ""
    title: str = ""
    selftext_excerpt: str = ""
    top_comments_excerpt: Tuple[str, ...] = ()
    raw_topic_text: str = ""
    controversy: float = 0.0
    emotional_load: float = 0.0
    interpersonalness: float = 0.0
    evidentiality: float = 0.0
    engagement_baitness: float = 0.0
    public_vs_personal: float = 0.5
    safety_risk_hint: float = 0.0
    coarse_category: str = ""

    @property
    def topic_category(self) -> str:
        if self.coarse_category:
            return self.coarse_category
        if self.source_subreddit:
            return self.source_subreddit
        return "uncategorized"

    def latent_axes(self) -> Dict[str, float]:
        return {
            "controversy": float(self.controversy),
            "emotional_load": float(self.emotional_load),
            "interpersonalness": float(self.interpersonalness),
            "evidentiality": float(self.evidentiality),
            "engagement_baitness": float(self.engagement_baitness),
            "public_vs_personal": float(self.public_vs_personal),
            "safety_risk_hint": float(self.safety_risk_hint),
        }


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str


DEFAULT_TOPICS: Tuple[Topic, ...] = ()


def default_topic_bank_path() -> str:
    return str(Path(__file__).with_name("topic_bank.sampled.jsonl"))


def default_trait_space() -> Dict[str, Any]:
    return {
        "dimensions": [
            {
                "name": "stance",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": "generally question or push back on the main idea",
                "prompt_positive": "generally support or agree with the main idea",
            },
            {
                "name": "sociality",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": "discuss abstractly, structurally, and evidence-seekingly",
                "prompt_positive": "make the topic interpersonal, identity-relevant, and close to everyday users",
            },
            {
                "name": "risk_tolerance",
                "min": -1.0,
                "max": 1.0,
                "prompt_negative": "avoid reckless escalation and prefer broad acceptability",
                "prompt_positive": "optimize for attention, reaction, and conversational pull",
            },
        ],
        "sampling": {
            "method": "latin_hypercube",
            "n_agents": 96,
            "seed": 42,
        },
        "evaluator_sampling": {
            "method": "latin_hypercube",
            "n_evaluators": 8,
            "seed": 7,
        },
        "discretization": {
            "enabled": True,
            "bins": {
                "stance": [-0.33, 0.33],
                "sociality": [-0.33, 0.33],
                "risk_tolerance": [-0.33, 0.33],
            },
        },
    }


@dataclass
class LogGenerationConfig:
    post_model: ModelSpec = field(default_factory=lambda: ModelSpec("openai", "gpt-5.4-mini"))
    eval_model: ModelSpec = field(default_factory=lambda: ModelSpec("openai", "gpt-5.4-mini"))
    audit_model: ModelSpec = field(default_factory=lambda: ModelSpec("openai", "gpt-5.4-mini"))
    trait_space: Optional[Dict[str, Any]] = field(default_factory=default_trait_space)
    model_heterogeneity: Optional[Dict[str, Any]] = None
    temperature_post: float = 0.7
    temperature_eval: float = 0.1
    agents_per_trait_combo: int = 12
    n_timesteps: int = 200
    topics: Tuple[Topic, ...] = field(default_factory=lambda: DEFAULT_TOPICS)
    topic_bank_path: str = field(default_factory=default_topic_bank_path)
    topic_sample_size: int = 0
    regenerate_content_each_timestep: bool = True
    run_label: str = ""
    api_max_workers: int = 4
    post_max_workers: Optional[int] = None
    eval_max_workers: Optional[int] = None
    audit_max_workers: Optional[int] = None
    api_timeout_sec: float = 120.0
    api_max_retries: int = 4
    sleep_between_calls_sec: float = 0.0
    random_seed: int = 42
    save_full_text: bool = True
    save_response_text: bool = True
    save_eval_model_metadata: bool = True
    save_audit_model_metadata: bool = True
    batch_evaluations_across_panel: bool = True


class OpenAIWrapper:
    def __init__(self, timeout_sec: float = 120.0, max_retries: int = 4) -> None:
        from openai import OpenAI

        openai_key = os.environ.get("OPENAI_API_KEY")
        together_key = os.environ.get("TOGETHER_API_KEY")

        self.timeout_sec = float(timeout_sec)
        self.max_retries = int(max_retries)
        self.openai_client: Optional[OpenAI] = None
        self.together_client: Optional[OpenAI] = None

        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key, timeout=self.timeout_sec)
        if together_key:
            self.together_client = OpenAI(
                api_key=together_key,
                base_url="https://api.together.xyz/v1",
                timeout=self.timeout_sec,
            )

    def _get_client(self, provider: str) -> Any:
        if provider == "openai":
            if self.openai_client is None:
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            return self.openai_client
        if provider == "together":
            if self.together_client is None:
                raise EnvironmentError("TOGETHER_API_KEY is not set.")
            return self.together_client
        raise ValueError(f"Unsupported provider: {provider}")

    def _chat_create(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        client = self._get_client(provider)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if provider == "openai":
                    request_kwargs["max_completion_tokens"] = max_tokens
                else:
                    request_kwargs["max_tokens"] = max_tokens
                return client.chat.completions.create(
                    **request_kwargs,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0 ** attempt, 8.0))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Chat completion failed without an exception.")

    def generate_post(
        self,
        spec: ModelSpec,
        trait: Any,
        topic: str,
        temperature: float,
        trait_space_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._poster_prompt(trait, topic, trait_space_cfg=trait_space_cfg)
        response = self._chat_create(
            provider=spec.provider,
            model=spec.model,
            messages=[
                {"role": "system", "content": "You are a social platform user. Stay in character and write exactly one short post."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=temperature,
        )
        raw = self._extract_text(response)
        text = raw.strip()
        return {
            "text": text,
            "prompt": prompt,
            "raw_response_text": raw,
            "provider": spec.provider,
            "model": spec.model,
        }

    def evaluate_post(
        self,
        spec: ModelSpec,
        evaluator_trait: Any,
        topic: str,
        post_text: str,
        temperature: float,
        trait_space_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._evaluator_prompt(evaluator_trait, topic, post_text, trait_space_cfg=trait_space_cfg)
        response = self._chat_create(
            provider=spec.provider,
            model=spec.model,
            messages=[
                {"role": "system", "content": "You are a platform user evaluating one post. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=220,
            temperature=temperature,
        )
        raw = self._extract_text(response).strip()
        parsed, parse_error = self._try_parse_json_like_response(raw)
        parsed["_meta"] = {
            "prompt": prompt,
            "raw_response_text": raw,
            "provider": spec.provider,
            "model": spec.model,
            "parse_error": parse_error,
        }
        return parsed

    def evaluate_post_panel(
        self,
        spec: ModelSpec,
        evaluators: Sequence[Evaluator],
        topic: str,
        post_text: str,
        temperature: float,
        trait_space_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._panel_evaluator_prompt(evaluators, topic, post_text, trait_space_cfg=trait_space_cfg)
        response = self._chat_create(
            provider=spec.provider,
            model=spec.model,
            messages=[
                {"role": "system", "content": "You are simulating a panel of users evaluating one post. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=temperature,
        )
        raw = self._extract_text(response).strip()
        parsed, parse_error = self._try_parse_json_like_response(raw)
        parsed["_meta"] = {
            "prompt": prompt,
            "raw_response_text": raw,
            "provider": spec.provider,
            "model": spec.model,
            "parse_error": parse_error,
        }
        return parsed

    def audit_post(self, spec: ModelSpec, topic: str, post_text: str, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = self._audit_prompt(topic, post_text)
        response = self._chat_create(
            provider=spec.provider,
            model=spec.model,
            messages=[
                {"role": "system", "content": "You are a neutral content auditor. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=160,
            temperature=temperature,
        )
        raw = self._extract_text(response).strip()
        parsed, parse_error = self._try_parse_json_like_response(raw)
        parsed["_meta"] = {
            "prompt": prompt,
            "raw_response_text": raw,
            "provider": spec.provider,
            "model": spec.model,
            "parse_error": parse_error,
        }
        return parsed

    @staticmethod
    def _extract_text(response: Any) -> str:
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json_like_response(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw[start:end + 1])
            raise ValueError(f"Response was not valid JSON: {raw}")

    @classmethod
    def _try_parse_json_like_response(cls, raw: str) -> Tuple[Dict[str, Any], str]:
        try:
            parsed = cls._parse_json_like_response(raw)
            if not isinstance(parsed, dict):
                return {}, f"response JSON was not an object: {type(parsed).__name__}"
            return parsed, ""
        except Exception as exc:
            return {}, str(exc)

    @staticmethod
    def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _poster_prompt(trait: Any, topic: str, trait_space_cfg: Optional[Dict[str, Any]] = None) -> str:
        trait_block = render_trait_prompt_block(trait, trait_space_cfg)
        return f"""
You are writing a short post on a social platform.

Your behavioral profile is:
{trait_block}

Task:
Write exactly one short post about:
{topic}

Constraints:
- exactly one short paragraph
- no hashtags
- no emojis
- no bullet points
- stay consistent with your traits
- do not mention that you are an agent
""".strip()

    @staticmethod
    def _evaluator_prompt(trait: Any, topic: str, post_text: str, trait_space_cfg: Optional[Dict[str, Any]] = None) -> str:
        trait_block = render_trait_prompt_block(trait, trait_space_cfg)
        return f"""
You are a user evaluating one post on a social platform.

Your behavioral profile is:
{trait_block}

Topic:
{topic}

Post:
```\n{post_text}\n```

Return JSON only with this schema:
{{
  "like": float,
  "reply": float,
  "share": float,
  "follow": float,
  "quality": float,
  "safety": float
}}

Interpretation:
- like: probability you would like the post
- reply: probability you would reply to the post
- share: probability you would reshare / boost the post
- follow: probability you would follow the author
- quality: how clear, coherent, and useful the post feels
- safety: how appropriate and low-risk the post feels for general distribution
""".strip()

    @staticmethod
    def _audit_prompt(topic: str, post_text: str) -> str:
        return f"""
You are a neutral content auditor evaluating one social-platform post.

Topic:
{topic}

Post:
```\n{post_text}\n```

Return JSON only with this schema:
{{
  "safety": float,
  "quality": float,
  "reason": string
}}

Scoring guidance:
- safety: 0 to 1, where 1 means broadly acceptable and low-risk for general distribution
- quality: 0 to 1, where 1 means clear, coherent, useful, and non-spammy
- reason: one short sentence
""".strip()

    @staticmethod
    def _panel_evaluator_prompt(
        evaluators: Sequence[Evaluator],
        topic: str,
        post_text: str,
        trait_space_cfg: Optional[Dict[str, Any]] = None,
    ) -> str:
        evaluator_specs = []
        for evaluator in evaluators:
            evaluator_specs.append(
                {
                    "evaluator_id": evaluator.evaluator_id,
                    "trait": asdict(evaluator.trait),
                    "trait_text": verbalize_trait(evaluator.trait, trait_space_cfg),
                }
            )
        return f"""
You are simulating a panel of users evaluating one post on a social platform.

Topic:
{topic}

Post:
```\n{post_text}\n```

Evaluators:
{json.dumps(evaluator_specs, ensure_ascii=False, indent=2)}

Return JSON only with this schema:
{{
  "evaluations": [
    {{
      "evaluator_id": string,
      "like": float,
      "reply": float,
      "share": float,
      "follow": float,
      "quality": float,
      "safety": float
    }}
  ]
}}

Requirements:
- include exactly one object for each evaluator_id listed above
- preserve each evaluator's behavioral profile when scoring
- scores should be between 0 and 1
""".strip()


def parallel_map_ordered(items: Sequence[Any], fn, max_workers: int) -> List[Any]:
    if not items:
        return []
    worker_count = max(1, min(int(max_workers), len(items)))
    if worker_count <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(fn, items))


def use_continuous_traits(cfg: LogGenerationConfig) -> bool:
    return bool(getattr(cfg, "trait_space", None))


def _trait_space_dimensions(trait_space_cfg: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not trait_space_cfg:
        return []
    dims = trait_space_cfg.get("dimensions", [])
    return [dim for dim in dims if isinstance(dim, dict) and dim.get("name")]


def verbalize_trait(trait: Any, trait_space_cfg: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(trait, TraitProfile):
        return "\n".join(
            [
                f"- position: {trait.position}",
                f"- specialization: {trait.specialization}",
                f"- objective orientation: {trait.objective_orientation}",
                "",
                "Trait definitions:",
                "- affirming: generally support or agree with the main idea",
                "- skeptical: question, challenge, or push back on the main idea",
                "- analytic: discuss the topic in an abstract, structured, or evidence-seeking way",
                "- social: make the topic feel interpersonal, identity-relevant, or close to everyday users",
                "- safety-first: avoid reckless escalation, reward caution and broad acceptability",
                "- engagement-first: optimize for attention, reaction, and conversational pull",
            ]
        )

    if not isinstance(trait, ContinuousTraitProfile):
        return f"- trait: {trait}"

    lines: List[str] = []
    for dim in _trait_space_dimensions(trait_space_cfg):
        name = str(dim["name"])
        value = float(trait.coords.get(name, 0.0))
        neg = str(dim.get("prompt_negative", f"move in the negative {name} direction"))
        pos = str(dim.get("prompt_positive", f"move in the positive {name} direction"))

        if value <= -0.67:
            desc = f"strongly tends to {neg}"
        elif value <= -0.2:
            desc = f"somewhat tends to {neg}"
        elif value < 0.2:
            desc = "is relatively balanced on this dimension"
        elif value < 0.67:
            desc = f"somewhat tends to {pos}"
        else:
            desc = f"strongly tends to {pos}"

        label_text = ""
        if trait.labels and name in trait.labels:
            label_text = f", label={trait.labels[name]}"
        lines.append(f"- {name}: {value:.2f} ({desc}{label_text})")

    if not lines:
        for name, value in sorted(trait.coords.items()):
            lines.append(f"- {name}: {float(value):.2f}")
    return "\n".join(lines)


def render_trait_prompt_block(trait: Any, trait_space_cfg: Optional[Dict[str, Any]] = None) -> str:
    return verbalize_trait(trait, trait_space_cfg)


def discretize_trait(
    trait: ContinuousTraitProfile,
    discretization_cfg: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    if not discretization_cfg or not discretization_cfg.get("enabled", False):
        return {}
    out: Dict[str, str] = {}
    bins_cfg = discretization_cfg.get("bins", {})
    for dim_name, cutpoints in bins_cfg.items():
        if dim_name not in trait.coords:
            continue
        cuts = [float(point) for point in cutpoints]
        if len(cuts) < 2:
            continue
        x = float(trait.coords[dim_name])
        if x < cuts[0]:
            out[dim_name] = "low"
        elif x < cuts[1]:
            out[dim_name] = "mid"
        else:
            out[dim_name] = "high"
    return out


def _label_from_bins(value: float, cutpoints: Sequence[float], low_label: str, mid_label: str, high_label: str) -> str:
    cuts = [float(point) for point in cutpoints]
    if len(cuts) < 2:
        return mid_label
    if value < cuts[0]:
        return low_label
    if value < cuts[1]:
        return mid_label
    return high_label


def default_trait_labels(coords: Dict[str, float], discretization_cfg: Optional[Dict[str, Any]]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    bins_cfg = (discretization_cfg or {}).get("bins", {})
    for name, value in coords.items():
        cutpoints = bins_cfg.get(name, [-0.33, 0.33])
        if name == "stance":
            labels[f"{name}_bucket"] = _label_from_bins(float(value), cutpoints, "skeptical", "balanced", "affirming")
        elif name == "sociality":
            labels[f"{name}_bucket"] = _label_from_bins(float(value), cutpoints, "analytic", "balanced", "social")
        elif name == "risk_tolerance":
            labels[f"{name}_bucket"] = _label_from_bins(float(value), cutpoints, "safety-first", "balanced", "engagement-first")
        else:
            labels[f"{name}_bucket"] = _label_from_bins(float(value), cutpoints, "low", "mid", "high")
    return labels


def _sample_uniform_profiles(
    dims: Sequence[Dict[str, Any]],
    n_agents: int,
    rng: random.Random,
    discretization_cfg: Optional[Dict[str, Any]],
) -> List[ContinuousTraitProfile]:
    profiles: List[ContinuousTraitProfile] = []
    for _ in range(n_agents):
        coords: Dict[str, float] = {}
        for dim in dims:
            lo = float(dim["min"])
            hi = float(dim["max"])
            coords[str(dim["name"])] = rng.uniform(lo, hi)
        profiles.append(ContinuousTraitProfile(coords=coords, labels=default_trait_labels(coords, discretization_cfg)))
    return profiles


def _sample_latin_hypercube_profiles(
    dims: Sequence[Dict[str, Any]],
    n_agents: int,
    rng: random.Random,
    discretization_cfg: Optional[Dict[str, Any]],
) -> List[ContinuousTraitProfile]:
    per_dim_values: Dict[str, List[float]] = {}
    for dim in dims:
        name = str(dim["name"])
        lo = float(dim["min"])
        hi = float(dim["max"])
        width = hi - lo
        values: List[float] = []
        for idx in range(n_agents):
            u = (idx + rng.random()) / float(n_agents)
            values.append(lo + u * width)
        rng.shuffle(values)
        per_dim_values[name] = values

    profiles: List[ContinuousTraitProfile] = []
    for idx in range(n_agents):
        coords = {name: per_dim_values[name][idx] for name in per_dim_values}
        profiles.append(ContinuousTraitProfile(coords=coords, labels=default_trait_labels(coords, discretization_cfg)))
    return profiles


def sample_trait_profiles(
    trait_space_cfg: Dict[str, Any],
    sampling_cfg: Optional[Dict[str, Any]] = None,
    count_key: str = "n_agents",
) -> List[ContinuousTraitProfile]:
    dims = _trait_space_dimensions(trait_space_cfg)
    if not dims:
        raise ValueError("trait_space.dimensions must be non-empty when continuous traits are enabled.")
    sampling = sampling_cfg or trait_space_cfg.get("sampling", {})
    n_agents = int(sampling.get(count_key, 0))
    if n_agents <= 0:
        raise ValueError(f"trait sampling config must define a positive integer {count_key}.")
    seed = int(sampling.get("seed", 42))
    method = str(sampling.get("method", "uniform")).strip().lower()
    rng = random.Random(seed)
    discretization_cfg = trait_space_cfg.get("discretization")

    if method == "latin_hypercube":
        return _sample_latin_hypercube_profiles(dims, n_agents, rng, discretization_cfg)
    if method == "uniform":
        return _sample_uniform_profiles(dims, n_agents, rng, discretization_cfg)
    raise ValueError(f"Unsupported trait sampling method: {method}")


def _matches_trait_threshold(rule_when: Dict[str, Any], trait: ContinuousTraitProfile) -> bool:
    dim = str(rule_when["dimension"])
    if dim not in trait.coords:
        return False
    x = float(trait.coords[dim])
    if "gt" in rule_when and not (x > float(rule_when["gt"])):
        return False
    if "gte" in rule_when and not (x >= float(rule_when["gte"])):
        return False
    if "lt" in rule_when and not (x < float(rule_when["lt"])):
        return False
    if "lte" in rule_when and not (x <= float(rule_when["lte"])):
        return False
    return True


def _resolve_model_assignment(
    assignment_cfg: Dict[str, Any],
    trait: Any,
    default_model: ModelSpec,
) -> ModelSpec:
    if not assignment_cfg:
        return default_model
    fallback = coerce_model_spec(assignment_cfg.get("fallback_model", asdict(default_model)))
    method = str(assignment_cfg.get("method", "by_trait_threshold")).strip().lower()
    if method != "by_trait_threshold" or not isinstance(trait, ContinuousTraitProfile):
        return fallback

    for rule in assignment_cfg.get("rules", []):
        when = rule.get("when", {})
        if isinstance(when, dict) and when.get("dimension") and _matches_trait_threshold(when, trait):
            return coerce_model_spec(rule["model"])
    return fallback


def resolve_agent_post_model(agent_index: int, trait: Any, cfg: LogGenerationConfig) -> ModelSpec:
    del agent_index
    hetero = cfg.model_heterogeneity or {}
    if not hetero.get("enabled", False):
        return cfg.post_model
    return _resolve_model_assignment(hetero.get("post_assignment", {}), trait, cfg.post_model)


def resolve_evaluator_model(evaluator_index: int, trait: Any, cfg: LogGenerationConfig) -> ModelSpec:
    del evaluator_index
    hetero = cfg.model_heterogeneity or {}
    if not hetero.get("enabled", False):
        return cfg.eval_model
    return _resolve_model_assignment(hetero.get("eval_assignment", {}), trait, cfg.eval_model)


def build_agents_from_config(cfg: LogGenerationConfig) -> List[Agent]:
    trait_profiles = sample_trait_profiles(cfg.trait_space or {})
    agents: List[Agent] = []
    for i, trait in enumerate(trait_profiles):
        agents.append(
            Agent(
                agent_id=f"agent_{i}",
                trait=trait,
                post_model=resolve_agent_post_model(i, trait, cfg),
            )
        )
    return agents


def build_evaluators_from_config(cfg: LogGenerationConfig) -> List[Evaluator]:
    trait_space_cfg = cfg.trait_space or {}
    evaluator_sampling = trait_space_cfg.get("evaluator_sampling")
    if isinstance(evaluator_sampling, dict):
        trait_profiles = sample_trait_profiles(
            trait_space_cfg,
            sampling_cfg=evaluator_sampling,
            count_key="n_evaluators",
        )
    else:
        trait_profiles = sample_trait_profiles(trait_space_cfg)
    evaluators: List[Evaluator] = []
    for i, trait in enumerate(trait_profiles):
        evaluators.append(
            Evaluator(
                evaluator_id=f"eval_{i}",
                trait=trait,
                eval_model=resolve_evaluator_model(i, trait, cfg),
            )
        )
    return evaluators


def build_agents(k: int) -> List[Agent]:
    profiles = [
        TraitProfile(position, specialization, objective_orientation)
        for position in ["affirming", "skeptical"]
        for specialization in ["analytic", "social"]
        for objective_orientation in ["safety-first", "engagement-first"]
    ]
    out: List[Agent] = []
    for p_idx, profile in enumerate(profiles):
        for rep in range(k):
            out.append(Agent(agent_id=f"agent_t{p_idx}_k{rep}", trait=profile, post_model=None))
    return out


def build_evaluators() -> List[Evaluator]:
    profiles = [
        TraitProfile(position, specialization, objective_orientation)
        for position in ["affirming", "skeptical"]
        for specialization in ["analytic", "social"]
        for objective_orientation in ["safety-first", "engagement-first"]
    ]
    return [Evaluator(evaluator_id=f"eval_{i}", trait=p, eval_model=None) for i, p in enumerate(profiles)]


def make_context_id(run_id: int, timestep: int, topic: str) -> str:
    return sha256(f"{run_id}|{timestep}|{topic}".encode("utf-8")).hexdigest()[:24]


def compute_text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:24]


class LLMLogGenerator:
    def __init__(self, cfg: LogGenerationConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)
        self.client = OpenAIWrapper(timeout_sec=cfg.api_timeout_sec, max_retries=cfg.api_max_retries)
        if use_continuous_traits(cfg):
            self.agents = build_agents_from_config(cfg)
            self.evaluators = build_evaluators_from_config(cfg)
        else:
            self.agents = build_agents(cfg.agents_per_trait_combo)
            self.evaluators = build_evaluators()
        self.evaluator_index = {e.evaluator_id: e for e in self.evaluators}
        self.topic_schedule = self._build_topic_schedule()

    def _trait_discretization_cfg(self) -> Optional[Dict[str, Any]]:
        return (self.cfg.trait_space or {}).get("discretization")

    def _trait_payload(self, trait: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if isinstance(trait, ContinuousTraitProfile):
            payload["trait_coords"] = dict(trait.coords)
            payload["trait_bins"] = discretize_trait(trait, self._trait_discretization_cfg())
            if trait.labels:
                payload["trait_labels"] = dict(trait.labels)
        return payload

    def _should_batch_evaluations(self) -> bool:
        if not self.cfg.batch_evaluations_across_panel:
            return False
        first_spec: Optional[Tuple[str, str]] = None
        for evaluator in self.evaluators:
            spec = evaluator.eval_model or self.cfg.eval_model
            spec_key = (spec.provider, spec.model)
            if first_spec is None:
                first_spec = spec_key
                continue
            if spec_key != first_spec:
                return False
        return True

    def _worker_count(self, lane: str) -> int:
        lane_value = {
            "post": self.cfg.post_max_workers,
            "eval": self.cfg.eval_max_workers,
            "audit": self.cfg.audit_max_workers,
        }[lane]
        return int(lane_value or self.cfg.api_max_workers)

    def _build_topic_schedule(self) -> List[Topic]:
        scheduled = list(self.cfg.topics)
        if not scheduled and self.cfg.topic_bank_path:
            scheduled = list(load_topics_from_jsonl(self.cfg.topic_bank_path))
        if not scheduled:
            raise ValueError("topics must be non-empty, or topic_bank_path must point to a non-empty JSONL bank.")
        if self.cfg.topic_sample_size > 0 and self.cfg.topic_sample_size < len(scheduled):
            scheduled = self.rng.sample(scheduled, self.cfg.topic_sample_size)
        self.rng.shuffle(scheduled)
        return scheduled

    def generate_incremental(
        self,
        checkpoint_dir: str,
        resume: bool = True,
        reused_contents_by_timestep: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        eval_only: bool = False,
    ) -> Dict[str, Any]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        metadata = {
            "experiment_type": "llm_log_dataset",
            "config": self._config_to_jsonable(),
            "eval_only": eval_only,
            "reused_contents": reused_contents_by_timestep is not None,
            "agents": [{"agent_id": a.agent_id, "trait": asdict(a.trait)} for a in self.agents],
            "evaluators": [{"evaluator_id": e.evaluator_id, "trait": asdict(e.trait)} for e in self.evaluators],
        }
        save_json_atomic(metadata, os.path.join(checkpoint_dir, "metadata.json"))

        timestep_bundles: List[Dict[str, Any]] = []
        for t in range(self.cfg.n_timesteps):
            step_path = os.path.join(checkpoint_dir, f"timestep_{t:04d}.json")
            if resume and os.path.exists(step_path):
                try:
                    bundle = load_json(step_path)
                    if self._is_complete_timestep_bundle(bundle):
                        timestep_bundles.append(bundle)
                        print(f"[resume] timestep={t} -> using existing checkpoint", flush=True)
                        continue
                except Exception:
                    pass

            if reused_contents_by_timestep is not None:
                if t not in reused_contents_by_timestep:
                    raise ValueError(f"Missing reused contents for timestep={t}")
                contents = reused_contents_by_timestep[t]
            else:
                contents = self._generate_contents_for_timestep(t)
            evaluations, audits = self._generate_evaluations_for_timestep(t, contents)
            self._validate_timestep_bundle(t, contents, evaluations, audits)
            bundle = {
                "run_id": self.cfg.random_seed,
                "run_label": self.cfg.run_label,
                "timestep": t,
                "context_id": contents[0]["context_id"] if contents else "", ##each timestep gets a fresh context_id even for repeated topics
                "topic": contents[0]["topic"] if contents else "",
                "topic_category": contents[0]["topic_category"] if contents else "",
                "topic_source": contents[0]["topic_source"] if contents else "",
                "source_subreddit": contents[0]["source_subreddit"] if contents else "",
                "source_post_id": contents[0]["source_post_id"] if contents else "",
                "topic_index": contents[0]["topic_index"] if contents else t,
                "contents": contents,
                "evaluations": evaluations,
                "audits": audits,
                "complete": True,
            }
            save_json_atomic(bundle, step_path)
            timestep_bundles.append(bundle)
            print(f"[checkpoint] saved timestep={t}: {step_path}", flush=True)

        dataset = dict(metadata)
        ordered = sorted(timestep_bundles, key=lambda b: int(b["timestep"]))
        dataset["contents"] = [row for bundle in ordered for row in bundle["contents"]]
        dataset["evaluations"] = [row for bundle in ordered for row in bundle["evaluations"]]
        dataset["audits"] = [row for bundle in ordered for row in bundle["audits"]]
        return dataset

    def _config_to_jsonable(self) -> Dict[str, Any]:
        cfg = asdict(self.cfg)
        cfg["topics"] = [asdict(topic) for topic in self.cfg.topics]
        return cfg

    def _is_complete_timestep_bundle(self, bundle: Dict[str, Any]) -> bool:
        n_agents = len(self.agents)
        n_evaluators = len(self.evaluators)
        return (
            bool(bundle.get("complete"))
            and len(bundle.get("contents", [])) == n_agents
            and len(bundle.get("evaluations", [])) == n_agents * n_evaluators
            and len(bundle.get("audits", [])) == n_agents
        )

    def _validate_timestep_bundle(self, timestep: int, contents: List[Dict[str, Any]], evaluations: List[Dict[str, Any]], audits: List[Dict[str, Any]]) -> None:
        expected_contents = len(self.agents)
        expected_evals = len(self.agents) * len(self.evaluators)
        expected_audits = len(self.agents)
        if len(contents) != expected_contents:
            raise ValueError(f"timestep={timestep}: expected {expected_contents} contents, got {len(contents)}")
        if len(evaluations) != expected_evals:
            raise ValueError(f"timestep={timestep}: expected {expected_evals} evaluations, got {len(evaluations)}")
        if len(audits) != expected_audits:
            raise ValueError(f"timestep={timestep}: expected {expected_audits} audits, got {len(audits)}")
        context_ids = {row["context_id"] for row in contents}
        if len(context_ids) != 1:
            raise ValueError(f"timestep={timestep}: expected exactly one context_id, got {len(context_ids)}")
        if any(not row.get("topic_category") for row in contents):
            raise ValueError(f"timestep={timestep}: missing topic_category in contents")

    def _generate_contents_for_timestep(self, timestep: int) -> List[Dict[str, Any]]:
        topic_index = timestep % len(self.topic_schedule)
        topic_obj = self.topic_schedule[topic_index]
        context_id = make_context_id(self.cfg.random_seed, timestep, topic_obj.text)
        tasks = [(timestep, agent, topic_obj, topic_index, context_id) for agent in self.agents]

        def worker(task: Tuple[int, Agent, Topic, int, str]) -> Dict[str, Any]:
            t, agent, topic, t_idx, c_id = task
            model_spec = agent.post_model or self.cfg.post_model
            result = self.client.generate_post(
                model_spec,
                agent.trait,
                topic.text,
                temperature=self.cfg.temperature_post,
                trait_space_cfg=self.cfg.trait_space,
            )
            if self.cfg.sleep_between_calls_sec > 0:
                time.sleep(self.cfg.sleep_between_calls_sec)
            row = {
                "run_id": self.cfg.random_seed,
                "run_label": self.cfg.run_label,
                "timestep": t,
                "context_id": c_id,
                "topic": topic.text,
                "topic_category": topic.topic_category,
                "topic_index": t_idx,
                "topic_source": topic.source,
                "source_subreddit": topic.source_subreddit,
                "source_post_id": topic.source_post_id,
                "normalized_from": topic.normalized_from,
                "raw_topic_text": topic.raw_topic_text,
                "topic_title": topic.title,
                "selftext_excerpt": topic.selftext_excerpt,
                "top_comments_excerpt": list(topic.top_comments_excerpt),
                "controversy": topic.controversy,
                "emotional_load": topic.emotional_load,
                "interpersonalness": topic.interpersonalness,
                "evidentiality": topic.evidentiality,
                "engagement_baitness": topic.engagement_baitness,
                "public_vs_personal": topic.public_vs_personal,
                "safety_risk_hint": topic.safety_risk_hint,
                "agent_id": agent.agent_id,
                "content_text": result["text"],
                "content_text_hash": compute_text_hash(result["text"]),
                "trait": asdict(agent.trait),
                "post_provider": result["provider"],
                "post_model": result["model"],
            }
            row.update(self._trait_payload(agent.trait))
            if self.cfg.save_response_text:
                row["post_raw_response_text"] = result["raw_response_text"]
            if self.cfg.save_full_text:
                row["post_prompt"] = result["prompt"]
            return row

        return parallel_map_ordered(tasks, worker, self._worker_count("post"))

    def _generate_evaluations_for_timestep(self, timestep: int, contents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        content_map = {row["agent_id"]: row for row in contents}
        panel_tasks: List[Any] = []
        audit_tasks: List[str] = []
        batch_evaluations = self._should_batch_evaluations()
        for agent in self.agents:
            if batch_evaluations:
                panel_tasks.append(agent.agent_id)
            else:
                for evaluator in self.evaluators:
                    panel_tasks.append((agent.agent_id, evaluator.evaluator_id))
            audit_tasks.append(agent.agent_id)

        def single_eval_row(content: Dict[str, Any], evaluator: Evaluator, result: Dict[str, Any]) -> Dict[str, Any]:
            row = {
                "run_id": self.cfg.random_seed,
                "run_label": self.cfg.run_label,
                "timestep": timestep,
                "context_id": content["context_id"],
                "agent_id": content["agent_id"],
                "content_text_hash": content["content_text_hash"],
                "evaluator_id": evaluator.evaluator_id,
                "evaluator_trait": asdict(evaluator.trait),
                "agent_trait": content.get("trait"),
                "topic": content["topic"],
                "topic_category": content["topic_category"],
                "topic_index": content["topic_index"],
                "topic_source": content["topic_source"],
                "source_subreddit": content["source_subreddit"],
                "source_post_id": content["source_post_id"],
                "controversy": content["controversy"],
                "emotional_load": content["emotional_load"],
                "interpersonalness": content["interpersonalness"],
                "evidentiality": content["evidentiality"],
                "engagement_baitness": content["engagement_baitness"],
                "public_vs_personal": content["public_vs_personal"],
                "safety_risk_hint": content["safety_risk_hint"],
                "eval_provider": result.get("_meta", {}).get("provider", self.cfg.eval_model.provider),
                "eval_model": result.get("_meta", {}).get("model", self.cfg.eval_model.model),
                "like": self.client._coerce_float(result.get("like")),
                "reply": self.client._coerce_float(result.get("reply")),
                "share": self.client._coerce_float(result.get("share")),
                "follow": self.client._coerce_float(result.get("follow")),
                "quality": self.client._coerce_float(result.get("quality")),
                "safety": self.client._coerce_float(result.get("safety")),
            }
            if "trait_coords" in content:
                row["agent_trait_coords"] = dict(content["trait_coords"])
            if "trait_bins" in content:
                row["agent_trait_bins"] = dict(content["trait_bins"])
            if "trait_labels" in content:
                row["agent_trait_labels"] = dict(content["trait_labels"])
            if isinstance(evaluator.trait, ContinuousTraitProfile):
                row["evaluator_trait_coords"] = dict(evaluator.trait.coords)
                row["evaluator_trait_bins"] = discretize_trait(evaluator.trait, self._trait_discretization_cfg())
                if evaluator.trait.labels:
                    row["evaluator_trait_labels"] = dict(evaluator.trait.labels)
            if self.cfg.save_eval_model_metadata:
                row["eval_prompt"] = result.get("_meta", {}).get("prompt", "")
                row["eval_raw_response_text"] = result.get("_meta", {}).get("raw_response_text", "")
                row["eval_parse_error"] = result.get("_meta", {}).get("parse_error", "")
            return row

        def panel_worker_unbatched(task: Tuple[str, str]) -> Dict[str, Any]:
            agent_id, evaluator_id = task
            content = content_map[agent_id]
            evaluator = self.evaluator_index[evaluator_id]
            model_spec = evaluator.eval_model or self.cfg.eval_model
            result = self.client.evaluate_post(
                model_spec,
                evaluator.trait,
                str(content["topic"]),
                str(content["content_text"]),
                temperature=self.cfg.temperature_eval,
                trait_space_cfg=self.cfg.trait_space,
            )
            if self.cfg.sleep_between_calls_sec > 0:
                time.sleep(self.cfg.sleep_between_calls_sec)
            return single_eval_row(content, evaluator, result)

        def panel_worker_batched(agent_id: str) -> List[Dict[str, Any]]:
            content = content_map[agent_id]
            shared_spec = self.evaluators[0].eval_model or self.cfg.eval_model
            result = self.client.evaluate_post_panel(
                shared_spec,
                self.evaluators,
                str(content["topic"]),
                str(content["content_text"]),
                temperature=self.cfg.temperature_eval,
                trait_space_cfg=self.cfg.trait_space,
            )
            if self.cfg.sleep_between_calls_sec > 0:
                time.sleep(self.cfg.sleep_between_calls_sec)

            raw_evaluations = result.get("evaluations")
            meta = result.get("_meta", {})
            parsed_by_evaluator: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw_evaluations, list):
                for item in raw_evaluations:
                    if isinstance(item, dict) and item.get("evaluator_id"):
                        parsed_by_evaluator[str(item["evaluator_id"])] = item

            rows: List[Dict[str, Any]] = []
            parse_error = str(meta.get("parse_error", ""))
            for evaluator in self.evaluators:
                evaluator_result = dict(parsed_by_evaluator.get(evaluator.evaluator_id, {}))
                evaluator_spec = evaluator.eval_model or self.cfg.eval_model
                evaluator_result["_meta"] = {
                    "prompt": meta.get("prompt", ""),
                    "raw_response_text": meta.get("raw_response_text", ""),
                    "provider": meta.get("provider", evaluator_spec.provider),
                    "model": meta.get("model", evaluator_spec.model),
                    "parse_error": parse_error or (
                        "" if evaluator.evaluator_id in parsed_by_evaluator else f"missing evaluator_id={evaluator.evaluator_id} in batched response"
                    ),
                }
                rows.append(single_eval_row(content, evaluator, evaluator_result))
            return rows

        def audit_worker(agent_id: str) -> Dict[str, Any]:
            content = content_map[agent_id]
            result = self.client.audit_post(
                self.cfg.audit_model,
                str(content["topic"]),
                str(content["content_text"]),
                temperature=0.0,
            )
            if self.cfg.sleep_between_calls_sec > 0:
                time.sleep(self.cfg.sleep_between_calls_sec)
            row = {
                "run_id": self.cfg.random_seed,
                "run_label": self.cfg.run_label,
                "timestep": timestep,
                "context_id": content["context_id"],
                "agent_id": agent_id,
                "content_text_hash": content["content_text_hash"],
                "topic": content["topic"],
                "topic_category": content["topic_category"],
                "topic_index": content["topic_index"],
                "topic_source": content["topic_source"],
                "source_subreddit": content["source_subreddit"],
                "source_post_id": content["source_post_id"],
                "controversy": content["controversy"],
                "emotional_load": content["emotional_load"],
                "interpersonalness": content["interpersonalness"],
                "evidentiality": content["evidentiality"],
                "engagement_baitness": content["engagement_baitness"],
                "public_vs_personal": content["public_vs_personal"],
                "safety_risk_hint": content["safety_risk_hint"],
                "auditor_id": "neutral_auditor",
                "audit_provider": result.get("_meta", {}).get("provider", self.cfg.audit_model.provider),
                "audit_model": result.get("_meta", {}).get("model", self.cfg.audit_model.model),
                "safety": self.client._coerce_float(result.get("safety")),
                "quality": self.client._coerce_float(result.get("quality")),
                "reason": result.get("reason", ""),
            }
            if "trait" in content:
                row["agent_trait"] = content["trait"]
            if "trait_coords" in content:
                row["trait_coords"] = dict(content["trait_coords"])
            if "trait_bins" in content:
                row["trait_bins"] = dict(content["trait_bins"])
            if "trait_labels" in content:
                row["trait_labels"] = dict(content["trait_labels"])
            if self.cfg.save_audit_model_metadata:
                row["audit_prompt"] = result.get("_meta", {}).get("prompt", "")
                row["audit_raw_response_text"] = result.get("_meta", {}).get("raw_response_text", "")
                row["audit_parse_error"] = result.get("_meta", {}).get("parse_error", "")
            return row

        if batch_evaluations:
            eval_row_groups = parallel_map_ordered(panel_tasks, panel_worker_batched, self._worker_count("eval"))
            eval_rows = [row for group in eval_row_groups for row in group]
        else:
            eval_rows = parallel_map_ordered(panel_tasks, panel_worker_unbatched, self._worker_count("eval"))

        return (
            eval_rows,
            parallel_map_ordered(audit_tasks, audit_worker, self._worker_count("audit")),
        )


def save_json_atomic(obj: Dict[str, Any], path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_topic_row(row: Dict[str, Any]) -> Topic:
    normalized = dict(row)
    if "category" in normalized and "coarse_category" not in normalized:
        normalized["coarse_category"] = normalized.pop("category")
    if "normalized_prompt" in normalized and "text" not in normalized:
        normalized["text"] = normalized.pop("normalized_prompt")
    if "raw_topic_text" not in normalized:
        normalized["raw_topic_text"] = normalized.get("normalized_from", normalized.get("title", normalized.get("text", "")))
    if "title" not in normalized:
        normalized["title"] = normalized.get("normalized_from", "")
    top_comments = normalized.get("top_comments_excerpt", ())
    if isinstance(top_comments, list):
        normalized["top_comments_excerpt"] = tuple(str(item) for item in top_comments)
    elif isinstance(top_comments, str):
        normalized["top_comments_excerpt"] = (top_comments,)
    elif top_comments is None:
        normalized["top_comments_excerpt"] = ()
    return Topic(**normalized)


def load_topics_from_jsonl(path: str) -> Tuple[Topic, ...]:
    rows: List[Topic] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_number}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Invalid topic row in {path} line {line_number}: expected object.")
            rows.append(_coerce_topic_row(parsed))
    return tuple(rows)


def group_contents_by_timestep(dataset: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in dataset.get("contents", []):
        t = int(row["timestep"])
        grouped.setdefault(t, []).append(row)
    return grouped


def parse_overrides(raw: str) -> Dict[str, Any]:
    return {} if not raw else json.loads(raw)


def coerce_model_spec(value: Any, default_provider: str = "openai") -> ModelSpec:
    if isinstance(value, ModelSpec):
        return value
    if isinstance(value, str):
        return ModelSpec(provider=default_provider, model=value)
    if isinstance(value, dict):
        return ModelSpec(
            provider=str(value.get("provider", default_provider)),
            model=str(value["model"]),
        )
    raise TypeError(f"Unsupported model spec: {value!r}")


def normalize_config_dict(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(cfg_dict)
    legacy_to_new = {
        "model_post": "post_model",
        "model_eval": "eval_model",
        "model_audit": "audit_model",
    }
    for legacy_key, new_key in legacy_to_new.items():
        if legacy_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized.pop(legacy_key)
        else:
            normalized.pop(legacy_key, None)

    for key in ("post_model", "eval_model", "audit_model"):
        if key in normalized:
            normalized[key] = coerce_model_spec(normalized[key])

    heterogeneity = normalized.get("model_heterogeneity")
    if isinstance(heterogeneity, dict):
        for assignment_key in ("post_assignment", "eval_assignment", "audit_assignment"):
            assignment = heterogeneity.get(assignment_key)
            if not isinstance(assignment, dict):
                continue
            if "fallback_model" in assignment:
                assignment["fallback_model"] = coerce_model_spec(assignment["fallback_model"])
            rules = assignment.get("rules", [])
            if isinstance(rules, list):
                for rule in rules:
                    if isinstance(rule, dict) and "model" in rule:
                        rule["model"] = coerce_model_spec(rule["model"])

    if "topics" in normalized and normalized["topics"] and isinstance(normalized["topics"][0], dict):
        normalized["topics"] = tuple(_coerce_topic_row(row) for row in normalized["topics"])

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="llm_log_dataset.json")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--config_json", default="")
    parser.add_argument("--reuse_contents_from", default="")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    cfg_dict = asdict(LogGenerationConfig())
    cfg_dict.update(parse_overrides(args.config_json))
    cfg_dict = normalize_config_dict(cfg_dict)
    cfg = LogGenerationConfig(**cfg_dict)
    if not cfg.regenerate_content_each_timestep:
        raise ValueError("regenerate_content_each_timestep must be true for theory-valid log collection.")

    generator = LLMLogGenerator(cfg)
    checkpoint_dir = args.checkpoint_dir or f"{args.output}.checkpoints"
    reused_contents_by_timestep = None
    if args.reuse_contents_from:
        base_dataset = load_json(args.reuse_contents_from)
        reused_contents_by_timestep = group_contents_by_timestep(base_dataset)

    dataset = generator.generate_incremental(
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        reused_contents_by_timestep=reused_contents_by_timestep,
        eval_only=args.eval_only,
    )
    save_json_atomic(dataset, args.output)
    print(f"Saved log dataset to {args.output}")
    print(f"Saved timestep checkpoints to {checkpoint_dir}")


if __name__ == "__main__":
    main()
