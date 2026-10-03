#!/usr/bin/env bash

set -euo pipefail

tb_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tb_script_path="${tb_script_dir}/$(basename "${BASH_SOURCE[0]}")"
cd "${tb_script_dir}"

tb_python="${PYTHON_BIN:-python3}"

# Replicates 1--5 use these population seeds in the existing experiments.
tb_seeds="${SEEDS:-11 22 33 44 55}"
tb_n_timesteps=100
tb_n_agents=100
tb_n_evaluators=8

# Selected experiment queues run independently.
tb_social_max_parallel="${SOCIAL_8D_MAX_PARALLEL_RUNS:-4}"
tb_social_api_workers="${SOCIAL_8D_API_MAX_WORKERS:-1}"
tb_llama_max_parallel="${LLAMA_ONLY_MAX_PARALLEL_RUNS:-2}"
tb_llama_api_workers="${LLAMA_ONLY_API_MAX_WORKERS:-8}"
tb_tool_max_parallel="${TOOL_MAX_PARALLEL_RUNS:-1}"
tb_heterogeneity_max_parallel="${HETEROGENEITY_MAX_PARALLEL_RUNS:-2}"
tb_heterogeneity_api_workers="${HETEROGENEITY_API_MAX_WORKERS:-8}"
tb_post_max_tokens="${POST_MAX_TOKENS:-200}"
tb_tool_n_agents="${TOOL_N_AGENTS:-96}"
tb_tool_n_tasks="${TOOL_N_TASKS:-100}"
tb_tool_api_workers="${TOOL_API_MAX_WORKERS:-4}"

# Keep exactly one descriptive word after runs_testbed_.
tb_social_output_dir="${SOCIAL_8D_OUTPUT_DIR:-runs_testbed_8d}"
tb_social_16d_output_dir="${SOCIAL_16D_OUTPUT_DIR:-runs_testbed_16d}"
tb_llama_output_dir="${LLAMA_ONLY_OUTPUT_DIR:-runs_testbed_llama}"
tb_tool_output_dir="${TOOL_OUTPUT_DIR:-runs_testbed_tool}"
tb_heterogeneity_output_dir="${HETEROGENEITY_OUTPUT_DIR:-runs_testbed_heterogeneity}"

# Space-separated queue names.
tb_experiments_raw="${EXPERIMENTS:-social_16d llama_only}"
read -r -a tb_experiments <<< "${tb_experiments_raw}"

tb_dry_run="${DRY_RUN:-0}"
tb_mode="launch"
tb_worker_experiment=""

usage() {
  cat <<'USAGE'
Usage:
  ./run_testbeds.sh
  ./run_testbeds.sh --dry-run

Environment overrides:
  EXPERIMENTS="social_16d llama_only"
  SEEDS="11 22 33 44 55"
  SOCIAL_16D_OUTPUT_DIR=...
  LLAMA_ONLY_OUTPUT_DIR=...
  SOCIAL_8D_OUTPUT_DIR=...
  TOOL_OUTPUT_DIR=...
  HETEROGENEITY_OUTPUT_DIR=...

The normal invocation starts one detached queue per experiment and returns.
Re-running it skips completed/running seeds and resumes incomplete checkpoints.
USAGE
}

case "${1:-}" in
  "")
    ;;
  --dry-run)
    tb_dry_run=1
    ;;
  --worker)
    tb_mode="worker"
    tb_worker_experiment="${2:-}"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file is missing: $1" >&2
    exit 2
  fi
}

require_file log_collect.py
require_file log_collect_heterogeneous.py
require_file log_collect_tool.py
require_file topic_bank.sampled.jsonl

for positive_integer in \
  "${tb_n_timesteps}" \
  "${tb_n_agents}" \
  "${tb_n_evaluators}" \
  "${tb_social_max_parallel}" \
  "${tb_social_api_workers}" \
  "${tb_llama_max_parallel}" \
  "${tb_llama_api_workers}" \
  "${tb_tool_max_parallel}" \
  "${tb_heterogeneity_max_parallel}" \
  "${tb_heterogeneity_api_workers}" \
  "${tb_post_max_tokens}" \
  "${tb_tool_n_agents}" \
  "${tb_tool_n_tasks}" \
  "${tb_tool_api_workers}"; do
  if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: ${positive_integer}" >&2
    exit 2
  fi
done

if [[ "${tb_dry_run}" != "0" && "${tb_dry_run}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1; got: ${tb_dry_run}" >&2
  exit 2
fi

read -r -a tb_seed_array <<< "${tb_seeds}"
if (( ${#tb_seed_array[@]} == 0 )); then
  echo "SEEDS must contain at least one population seed." >&2
  exit 2
fi

tb_seen_seeds=" "
for pop_seed in "${tb_seed_array[@]}"; do
  if [[ ! "${pop_seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid population seed: ${pop_seed}" >&2
    exit 2
  fi
  if [[ "${tb_seen_seeds}" == *" ${pop_seed} "* ]]; then
    echo "Duplicate population seed: ${pop_seed}" >&2
    exit 2
  fi
  tb_seen_seeds+="${pop_seed} "
done

if (( ${#tb_experiments[@]} == 0 )); then
  echo "EXPERIMENTS must contain at least one queue name." >&2
  exit 2
fi

tb_seen_experiments=" "
for tb_experiment in "${tb_experiments[@]}"; do
  case "${tb_experiment}" in
    social_16d|llama_only|social_8d|tool_bfcl|generator_heterogeneity)
      ;;
    *)
      echo "Invalid experiment queue: ${tb_experiment}" >&2
      exit 2
      ;;
  esac
  if [[ "${tb_seen_experiments}" == *" ${tb_experiment} "* ]]; then
    echo "Duplicate experiment queue: ${tb_experiment}" >&2
    exit 2
  fi
  tb_seen_experiments+="${tb_experiment} "
done

experiment_selected() {
  local requested="$1"
  local experiment
  for experiment in "${tb_experiments[@]}"; do
    [[ "${experiment}" == "${requested}" ]] && return 0
  done
  return 1
}

if experiment_selected generator_heterogeneity &&
   (( tb_n_agents % 3 != 0 )); then
  echo "N_AGENTS must be a multiple of 3 for the fully crossed three-model experiment; got ${tb_n_agents}." >&2
  exit 2
fi

if [[ "${tb_dry_run}" == "0" && "${tb_mode}" == "launch" ]]; then
  echo "Preflight: checking API credentials and experiment queues..."
  tb_missing_keys=()
  if experiment_selected social_16d ||
     experiment_selected social_8d ||
     experiment_selected tool_bfcl ||
     experiment_selected generator_heterogeneity; then
    [[ -n "${OPENAI_API_KEY:-}" ]] || tb_missing_keys+=("OPENAI_API_KEY")
  fi
  if experiment_selected llama_only ||
     experiment_selected generator_heterogeneity; then
    [[ -n "${TOGETHER_API_KEY:-}" ]] || tb_missing_keys+=("TOGETHER_API_KEY")
  fi
  if (( ${#tb_missing_keys[@]} != 0 )); then
    echo "ERROR: missing API key(s): ${tb_missing_keys[*]}" >&2
    echo "Set them in this shell, then run ./run_testbeds.sh again." >&2
    exit 2
  fi
  "${tb_python}" -c "import openai"
fi

experiment_output_dir() {
  case "$1" in
    social_16d) printf '%s\n' "${tb_social_16d_output_dir}" ;;
    llama_only) printf '%s\n' "${tb_llama_output_dir}" ;;
    social_8d) printf '%s\n' "${tb_social_output_dir}" ;;
    tool_bfcl) printf '%s\n' "${tb_tool_output_dir}" ;;
    generator_heterogeneity) printf '%s\n' "${tb_heterogeneity_output_dir}" ;;
    *)
      echo "Unknown experiment: $1" >&2
      return 2
      ;;
  esac
}

experiment_expected_steps() {
  case "$1" in
    social_16d|llama_only|social_8d|generator_heterogeneity) printf '%s\n' "${tb_n_timesteps}" ;;
    tool_bfcl) printf '%s\n' "${tb_tool_n_tasks}" ;;
    *)
      echo "Unknown experiment: $1" >&2
      return 2
      ;;
  esac
}

experiment_max_parallel() {
  case "$1" in
    social_16d|social_8d) printf '%s\n' "${tb_social_max_parallel}" ;;
    llama_only) printf '%s\n' "${tb_llama_max_parallel}" ;;
    tool_bfcl) printf '%s\n' "${tb_tool_max_parallel}" ;;
    generator_heterogeneity) printf '%s\n' "${tb_heterogeneity_max_parallel}" ;;
    *)
      echo "Unknown experiment: $1" >&2
      return 2
      ;;
  esac
}

experiment_collector() {
  case "$1" in
    social_16d|llama_only|social_8d) printf '%s\n' "log_collect.py" ;;
    tool_bfcl) printf '%s\n' "log_collect_tool.py" ;;
    generator_heterogeneity) printf '%s\n' "log_collect_heterogeneous.py" ;;
    *)
      echo "Unknown experiment: $1" >&2
      return 2
      ;;
  esac
}

seed_is_complete() {
  local experiment="$1"
  local pop_seed="$2"
  local output_dir expected_steps checkpoint_dir checkpoint_count

  output_dir="$(experiment_output_dir "${experiment}")"
  expected_steps="$(experiment_expected_steps "${experiment}")"
  checkpoint_dir="${output_dir}/run_popseed_${pop_seed}.checkpoints"

  [[ -s "${output_dir}/run_popseed_${pop_seed}.json" ]] || return 1
  [[ -d "${checkpoint_dir}" ]] || return 1

  checkpoint_count="$(
    find "${checkpoint_dir}" -maxdepth 1 -type f -name 'timestep_*.json' |
      wc -l |
      tr -d '[:space:]'
  )"
  [[ "${checkpoint_count}" == "${expected_steps}" ]]
}

pid_matches_seed_job() {
  local pid="$1"
  local experiment="$2"
  local pop_seed="$3"
  local output_dir collector command

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1

  output_dir="$(experiment_output_dir "${experiment}")"
  collector="$(experiment_collector "${experiment}")"
  command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  [[ "${command}" == *"${collector}"* &&
     "${command}" == *"${output_dir}/run_popseed_${pop_seed}"* ]]
}

seed_is_running() {
  local experiment="$1"
  local pop_seed="$2"
  local output_dir pid_file pid

  output_dir="$(experiment_output_dir "${experiment}")"
  pid_file="${output_dir}/run_popseed_${pop_seed}.pid"
  [[ -f "${pid_file}" ]] || return 1
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  pid_matches_seed_job "${pid}" "${experiment}" "${pop_seed}"
}

build_social_config() {
  local experiment="$1"
  local pop_seed="$2"

  "${tb_python}" - \
    "${experiment}" \
    "${pop_seed}" \
    "${tb_n_timesteps}" \
    "${tb_n_agents}" \
    "${tb_n_evaluators}" \
    "${tb_social_api_workers}" \
    "${tb_llama_api_workers}" \
    "${tb_heterogeneity_api_workers}" \
    "${tb_post_max_tokens}" <<'PY'
import json
import sys

(
    experiment,
    pop_seed_raw,
    n_timesteps_raw,
    n_agents_raw,
    n_evaluators_raw,
    social_workers_raw,
    llama_workers_raw,
    heterogeneity_workers_raw,
    post_max_tokens_raw,
) = sys.argv[1:]

pop_seed = int(pop_seed_raw)
n_timesteps = int(n_timesteps_raw)
n_agents = int(n_agents_raw)
n_evaluators = int(n_evaluators_raw)
social_workers = int(social_workers_raw)
llama_workers = int(llama_workers_raw)
heterogeneity_workers = int(heterogeneity_workers_raw)
post_max_tokens = int(post_max_tokens_raw)

dimensions = [
    {
        "name": "stance_valence",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "examine the topic from a critical or counter-argument perspective",
        "prompt_positive": "develop the strongest reasonable case in favor of the topic's main premise",
    },
    {
        "name": "social_framing",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "frame the topic in terms of systems, institutions, incentives, or abstract structures",
        "prompt_positive": "frame the topic in terms of everyday experience, relationships, identity, or interpersonal consequences",
    },
    {
        "name": "epistemic_style",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "reason impressionistically, using intuition, analogies, or informal judgment",
        "prompt_positive": "reason with explicit uncertainty, evidence standards, caveats, and testable distinctions",
    },
    {
        "name": "consensus_alignment",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "challenge prevailing assumptions, surface overlooked counterpoints, or argue against the position most readers likely hold",
        "prompt_positive": "build from widely shared premises, emphasize points likely to feel reasonable to most readers, and avoid needlessly contrarian framing",
    },
]

config = {
    "post_model": {"provider": "openai", "model": "gpt-5.4-mini"},
    "eval_model": {"provider": "openai", "model": "gpt-5.4-mini"},
    "audit_model": {"provider": "openai", "model": "gpt-5.4-mini"},
    "temperature_post": 0.7,
    "temperature_eval": 0.2,
    "api_max_workers": social_workers,
    "post_max_workers": social_workers,
    "eval_max_workers": social_workers,
    "audit_max_workers": social_workers,
    "api_timeout_sec": 240,
    "api_max_retries": 3,
    "random_seed": 42,
    "n_timesteps": n_timesteps,
    "trait_space": {
        "dimensions": dimensions,
        "sampling": {
            "method": "latin_hypercube",
            "n_agents": n_agents,
            "seed": pop_seed,
        },
        "evaluator_sampling": {
            "method": "latin_hypercube",
            "n_evaluators": n_evaluators,
            "seed": 1000 + pop_seed,
        },
        "discretization": {
            "enabled": True,
            "bins": {dimension["name"]: [-0.5, 0.5] for dimension in dimensions},
        },
    },
}

if experiment in {"social_8d", "social_16d"}:
    extra_dimensions = [
        {
            "name": "emotional_expressiveness",
            "min": -1.0,
            "max": 1.0,
            "prompt_negative": "use emotionally restrained and neutral language",
            "prompt_positive": "use emotionally expressive language and foreground affective reactions",
        },
        {
            "name": "temporal_orientation",
            "min": -1.0,
            "max": 1.0,
            "prompt_negative": "focus on immediate consequences and short-term tradeoffs",
            "prompt_positive": "focus on long-term consequences and delayed effects",
        },
        {
            "name": "cooperative_orientation",
            "min": -1.0,
            "max": 1.0,
            "prompt_negative": "adopt an adversarial or competitive conversational posture",
            "prompt_positive": "seek common ground and adopt a collaborative conversational posture",
        },
        {
            "name": "normative_basis",
            "min": -1.0,
            "max": 1.0,
            "prompt_negative": "evaluate claims mainly by their outcomes and practical consequences",
            "prompt_positive": "evaluate claims mainly through duties, rights, rules, and principles",
        },
    ]
    if experiment == "social_16d":
        extra_dimensions.extend(
            [
                {
                    "name": "communication_directness",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "communicate indirectly and diplomatically, softening blunt conclusions",
                    "prompt_positive": "communicate directly and explicitly, stating conclusions plainly",
                },
                {
                    "name": "linguistic_formality",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "use an informal, conversational register",
                    "prompt_positive": "use a formal, professional register",
                },
                {
                    "name": "explanatory_granularity",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "keep explanations concise and high-level",
                    "prompt_positive": "give detailed explanations with explicit intermediate mechanisms",
                },
                {
                    "name": "solution_orientation",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "prioritize diagnosis and interpretation over prescribing actions",
                    "prompt_positive": "prioritize concrete recommendations and actionable solutions",
                },
                {
                    "name": "novelty_orientation",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "favor conventional, familiar approaches",
                    "prompt_positive": "favor novel, unconventional approaches",
                },
                {
                    "name": "perspective_breadth",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "develop one focused perspective",
                    "prompt_positive": "synthesize multiple distinct perspectives",
                },
                {
                    "name": "causal_focus",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "focus on observed associations and descriptive patterns",
                    "prompt_positive": "focus on causal mechanisms, interventions, and counterfactuals",
                },
                {
                    "name": "risk_posture",
                    "min": -1.0,
                    "max": 1.0,
                    "prompt_negative": "take a cautious, precautionary, risk-averse stance",
                    "prompt_positive": "take an experimental, risk-tolerant stance while acknowledging downsides",
                },
            ]
        )
    config["trait_space"]["dimensions"].extend(extra_dimensions)
    bins = config["trait_space"]["discretization"]["bins"]
    for dimension in extra_dimensions:
        bins[dimension["name"]] = [-0.5, 0.5]

    config["run_label"] = (
        f"{len(config['trait_space']['dimensions'])}d_"
        f"n{n_agents}_eval{n_evaluators}_popseed_{pop_seed}"
    )
    for key in (
        "api_max_workers",
        "post_max_workers",
        "eval_max_workers",
        "audit_max_workers",
    ):
        config[key] = social_workers
elif experiment == "llama_only":
    llama_model = {
        "provider": "together",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    }
    config.update(
        {
            "post_model": llama_model,
            "eval_model": llama_model,
            "audit_model": llama_model,
            "run_label": (
                f"llama_only_n{n_agents}_eval{n_evaluators}_"
                f"popseed_{pop_seed}"
            ),
        }
    )
    for key in (
        "api_max_workers",
        "post_max_workers",
        "eval_max_workers",
        "audit_max_workers",
    ):
        config[key] = llama_workers
elif experiment == "generator_heterogeneity":
    config.update(
        {
            "run_label": f"generator_heterogeneity_popseed_{pop_seed}",
            "temperature_eval": 0.1,
            "post_max_tokens": post_max_tokens,
            "batch_evaluations_across_panel": True,
            "save_full_text": False,
            "save_response_text": False,
            "save_eval_model_metadata": False,
            "save_audit_model_metadata": False,
            "model_heterogeneity": {
                "enabled": True,
                "post_assignment": {
                    "method": "fully_crossed",
                    "models": [
                        {"provider": "openai", "model": "gpt-5.4-mini"},
                        {
                            "provider": "together",
                            "model": "deepseek-ai/DeepSeek-V4-Pro",
                        },
                        {
                            "provider": "together",
                            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        },
                    ],
                },
            },
        }
    )
    for key in (
        "api_max_workers",
        "post_max_workers",
        "eval_max_workers",
        "audit_max_workers",
    ):
        config[key] = heterogeneity_workers
else:
    raise ValueError(f"unknown experiment: {experiment}")

print(json.dumps(config, ensure_ascii=False, separators=(",", ":")))
PY
}

build_tool_config() {
  local pop_seed="$1"

  "${tb_python}" - \
    "${pop_seed}" \
    "${tb_tool_n_agents}" \
    "${tb_tool_n_tasks}" \
    "${tb_tool_api_workers}" <<'PY'
import json
import sys

pop_seed = int(sys.argv[1])
n_agents = int(sys.argv[2])
n_tasks = int(sys.argv[3])
api_workers = int(sys.argv[4])

dimensions = [
    {
        "name": "action_propensity",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "use a high action threshold: abstain rather than call a tool when the request is irrelevant or materially underspecified",
        "prompt_positive": "use a low action threshold: call an applicable tool whenever the user's likely requested action is sufficiently clear",
    },
    {
        "name": "intent_interpretation",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "interpret requests literally and avoid extending beyond their explicitly stated intent",
        "prompt_positive": "infer implicit intent from the full request and tool descriptions when the intended operation is reasonably clear",
    },
    {
        "name": "argument_policy",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "supply only arguments explicitly supported by the request; do not invent missing values",
        "prompt_positive": "use schema defaults and strongly implied values when doing so is necessary to complete the requested operation",
    },
    {
        "name": "orchestration_policy",
        "min": -1.0,
        "max": 1.0,
        "prompt_negative": "prefer minimal atomic invocation and avoid unnecessary batching",
        "prompt_positive": "emit all independent required calls together and use parallel orchestration when the request contains multiple operations",
    },
]

config = {
    "model": {"provider": "openai", "model": "gpt-5.4-mini"},
    "subset_path": "benchmarks/bfcl/tasks.jsonl",
    "n_tasks": n_tasks,
    "task_sample_seed": 42,
    "random_seed": 42,
    "run_label": f"tool_bfcl_popseed_{pop_seed}",
    "temperature": 0.7,
    "max_completion_tokens": 512,
    "api_max_workers": api_workers,
    "api_timeout_sec": 120,
    "api_max_retries": 4,
    "parallel_tool_calls": True,
    "save_prompt": True,
    "save_raw_response": True,
    "trait_space": {
        "dimensions": dimensions,
        "sampling": {
            "method": "latin_hypercube",
            "n_agents": n_agents,
            "seed": pop_seed,
        },
        "discretization": {
            "enabled": True,
            "bins": {dimension["name"]: [-0.5, 0.5] for dimension in dimensions},
        },
    },
}

print(json.dumps(config, ensure_ascii=False, separators=(",", ":")))
PY
}

validate_social_config() {
  local config_json="$1"
  local expected_dimensions="$2"
  local expected_agents="$3"
  local expected_evaluators="$4"

  "${tb_python}" -c '
import json
import sys

config = json.loads(sys.argv[1])
assert len(config["trait_space"]["dimensions"]) == int(sys.argv[2])
assert config["trait_space"]["sampling"]["n_agents"] == int(sys.argv[3])
assert config["trait_space"]["evaluator_sampling"]["n_evaluators"] == int(sys.argv[4])
' "${config_json}" "${expected_dimensions}" "${expected_agents}" "${expected_evaluators}"
}

validate_llama_config() {
  local config_json="$1"

  validate_social_config \
    "${config_json}" 4 "${tb_n_agents}" "${tb_n_evaluators}"
  "${tb_python}" -c '
import json
import sys

config = json.loads(sys.argv[1])
for key in ("post_model", "eval_model", "audit_model"):
    assert config[key]["provider"] == "together"
    assert config[key]["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
' "${config_json}"
}

validate_tool_config() {
  local config_json="$1"

  "${tb_python}" -c '
import json
import sys

config = json.loads(sys.argv[1])
assert len(config["trait_space"]["dimensions"]) == 4
assert config["trait_space"]["sampling"]["n_agents"] > 0
assert config["n_tasks"] > 0
' "${config_json}"
}

run_seed_job() {
  local experiment="$1"
  local pop_seed="$2"
  local output_dir config_json

  output_dir="$(experiment_output_dir "${experiment}")"
  echo "Starting experiment=${experiment}, pop_seed=${pop_seed}, output_dir=${output_dir}"

  case "${experiment}" in
    social_16d)
      config_json="$(build_social_config "${experiment}" "${pop_seed}")"
      validate_social_config \
        "${config_json}" 16 "${tb_n_agents}" "${tb_n_evaluators}"
      exec "${tb_python}" -u log_collect.py \
        --output "${output_dir}/run_popseed_${pop_seed}.json" \
        --checkpoint_dir "${output_dir}/run_popseed_${pop_seed}.checkpoints" \
        --config_json "${config_json}"
      ;;
    llama_only)
      config_json="$(build_social_config "${experiment}" "${pop_seed}")"
      validate_llama_config "${config_json}"
      exec "${tb_python}" -u log_collect.py \
        --output "${output_dir}/run_popseed_${pop_seed}.json" \
        --checkpoint_dir "${output_dir}/run_popseed_${pop_seed}.checkpoints" \
        --config_json "${config_json}"
      ;;
    social_8d)
      config_json="$(build_social_config "${experiment}" "${pop_seed}")"
      validate_social_config \
        "${config_json}" 8 "${tb_n_agents}" "${tb_n_evaluators}"
      exec "${tb_python}" -u log_collect.py \
        --output "${output_dir}/run_popseed_${pop_seed}.json" \
        --checkpoint_dir "${output_dir}/run_popseed_${pop_seed}.checkpoints" \
        --config_json "${config_json}"
      ;;
    tool_bfcl)
      config_json="$(build_tool_config "${pop_seed}")"
      validate_tool_config "${config_json}"
      exec "${tb_python}" -u log_collect_tool.py \
        --config_json "${config_json}" \
        --population_seed "${pop_seed}" \
        --output "${output_dir}/run_popseed_${pop_seed}.json" \
        --checkpoint_dir "${output_dir}/run_popseed_${pop_seed}.checkpoints"
      ;;
    generator_heterogeneity)
      config_json="$(build_social_config "${experiment}" "${pop_seed}")"
      validate_social_config \
        "${config_json}" 4 "${tb_n_agents}" "${tb_n_evaluators}"
      exec "${tb_python}" -u log_collect_heterogeneous.py \
        --output "${output_dir}/run_popseed_${pop_seed}.json" \
        --checkpoint_dir "${output_dir}/run_popseed_${pop_seed}.checkpoints" \
        --config_json "${config_json}"
      ;;
    *)
      echo "Unknown experiment: ${experiment}" >&2
      return 2
      ;;
  esac
}

print_seed_plan() {
  local experiment="$1"
  local pop_seed="$2"
  local output_dir expected_steps

  output_dir="$(experiment_output_dir "${experiment}")"
  expected_steps="$(experiment_expected_steps "${experiment}")"
  printf 'PLAN experiment=%s pop_seed=%s steps=%s output=%s checkpoints=%s\n' \
    "${experiment}" \
    "${pop_seed}" \
    "${expected_steps}" \
    "${output_dir}/run_popseed_${pop_seed}.json" \
    "${output_dir}/run_popseed_${pop_seed}.checkpoints"
}

remove_pid_file_if_matching() {
  local pid_file="$1"
  local expected_pid="$2"
  local current_pid

  [[ -f "${pid_file}" ]] || return 0
  current_pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ "${current_pid}" == "${expected_pid}" ]]; then
    rm -f "${pid_file}"
  fi
}

run_experiment_queue() {
  local experiment="$1"
  local output_dir max_parallel pop_seed pid pid_file index failed config_json
  local -a launched_pids=()
  local -a launched_seeds=()

  output_dir="$(experiment_output_dir "${experiment}")"
  max_parallel="$(experiment_max_parallel "${experiment}")"
  mkdir -p "${output_dir}"

  for pop_seed in "${tb_seed_array[@]}"; do
    if seed_is_complete "${experiment}" "${pop_seed}"; then
      echo "SKIP complete experiment=${experiment}, pop_seed=${pop_seed}"
      continue
    fi

    if seed_is_running "${experiment}" "${pop_seed}"; then
      echo "SKIP running experiment=${experiment}, pop_seed=${pop_seed}"
      continue
    fi

    if [[ "${tb_dry_run}" == "1" ]]; then
      case "${experiment}" in
        social_16d)
          config_json="$(build_social_config "${experiment}" "${pop_seed}")"
          validate_social_config \
            "${config_json}" 16 "${tb_n_agents}" "${tb_n_evaluators}"
          ;;
        llama_only)
          config_json="$(build_social_config "${experiment}" "${pop_seed}")"
          validate_llama_config "${config_json}"
          ;;
        social_8d)
          config_json="$(build_social_config "${experiment}" "${pop_seed}")"
          validate_social_config \
            "${config_json}" 8 "${tb_n_agents}" "${tb_n_evaluators}"
          ;;
        tool_bfcl)
          config_json="$(build_tool_config "${pop_seed}")"
          validate_tool_config "${config_json}"
          ;;
        generator_heterogeneity)
          config_json="$(build_social_config "${experiment}" "${pop_seed}")"
          validate_social_config \
            "${config_json}" 4 "${tb_n_agents}" "${tb_n_evaluators}"
          ;;
      esac
      print_seed_plan "${experiment}" "${pop_seed}"
      continue
    fi

    while (( $(jobs -rp | wc -l | tr -d '[:space:]') >= max_parallel )); do
      sleep 2
    done

    run_seed_job "${experiment}" "${pop_seed}" \
      > "${output_dir}/run_popseed_${pop_seed}.log" 2>&1 &
    pid=$!
    pid_file="${output_dir}/run_popseed_${pop_seed}.pid"
    printf '%s\n' "${pid}" > "${pid_file}"
    launched_pids+=("${pid}")
    launched_seeds+=("${pop_seed}")
    echo "QUEUED experiment=${experiment}, pop_seed=${pop_seed}, pid=${pid}"
  done

  if [[ "${tb_dry_run}" == "1" ]]; then
    return 0
  fi

  failed=0
  for index in "${!launched_pids[@]}"; do
    pid="${launched_pids[${index}]}"
    pop_seed="${launched_seeds[${index}]}"
    pid_file="${output_dir}/run_popseed_${pop_seed}.pid"
    if wait "${pid}"; then
      echo "DONE experiment=${experiment}, pop_seed=${pop_seed}"
    else
      echo "FAILED experiment=${experiment}, pop_seed=${pop_seed}; see ${output_dir}/run_popseed_${pop_seed}.log" >&2
      failed=1
    fi
    remove_pid_file_if_matching "${pid_file}" "${pid}"
  done

  if (( failed != 0 )); then
    return 1
  fi
  echo "QUEUE COMPLETE experiment=${experiment}, output_dir=${output_dir}"
}

supervisor_is_running() {
  local experiment="$1"
  local output_dir pid_file pid command

  output_dir="$(experiment_output_dir "${experiment}")"
  pid_file="${output_dir}/launcher.pid"
  [[ -f "${pid_file}" ]] || return 1

  pid="$(tr -d '[:space:]' < "${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  [[ "${command}" == *"$(basename "${tb_script_path}") --worker ${experiment}"* ]]
}

experiment_has_pending_seed() {
  local experiment="$1"
  local pop_seed

  for pop_seed in "${tb_seed_array[@]}"; do
    if ! seed_is_complete "${experiment}" "${pop_seed}" &&
       ! seed_is_running "${experiment}" "${pop_seed}"; then
      return 0
    fi
  done
  return 1
}

launch_experiment_supervisor() {
  local experiment="$1"
  local output_dir launcher_pid

  output_dir="$(experiment_output_dir "${experiment}")"
  mkdir -p "${output_dir}"

  if supervisor_is_running "${experiment}"; then
    echo "SKIP supervisor already running experiment=${experiment}"
    return 0
  fi

  if ! experiment_has_pending_seed "${experiment}"; then
    echo "SKIP no pending seeds experiment=${experiment}"
    return 0
  fi

  nohup "${tb_script_path}" --worker "${experiment}" \
    >> "${output_dir}/launcher.log" 2>&1 < /dev/null &
  launcher_pid=$!
  printf '%s\n' "${launcher_pid}" > "${output_dir}/launcher.pid"
  echo "STARTED supervisor experiment=${experiment}, pid=${launcher_pid}, log=${output_dir}/launcher.log"
}

if [[ "${tb_mode}" == "worker" ]]; then
  case "${tb_worker_experiment}" in
    social_16d|llama_only|social_8d|tool_bfcl|generator_heterogeneity)
      ;;
    *)
      echo "Invalid worker experiment: ${tb_worker_experiment}" >&2
      exit 2
      ;;
  esac
  run_experiment_queue "${tb_worker_experiment}"
  exit $?
fi

if [[ "${tb_dry_run}" == "1" ]]; then
  for tb_experiment in "${tb_experiments[@]}"; do
    run_experiment_queue "${tb_experiment}"
  done
  exit 0
fi

for tb_experiment in "${tb_experiments[@]}"; do
  launch_experiment_supervisor "${tb_experiment}"
done

echo "All testbed experiment queues have been submitted."
