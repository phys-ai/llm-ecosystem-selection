
#mkdir -p ./runs



##for pop_seed in 11 22 33 44 55 66 77 88; do
#for pop_seed in 99 110; do
#  python3 log_collect.py \
#    --output "runs/run_popseed_${pop_seed}.json" \
#    --checkpoint_dir "runs/run_popseed_${pop_seed}.checkpoints" \
#    --config_json "{
#      \"random_seed\": 42,
#      \"n_timesteps\": 200,
#      \"trait_space\": {
#        \"dimensions\": [
#          {\"name\": \"stance_valence\", \"min\": -1.0, \"max\": 1.0,
#           \"prompt_negative\": \"examine the topic from a critical or counter-argument perspective\",
#           \"prompt_positive\": \"develop the strongest reasonable case in favor of the topic's main premise\"},
#          {\"name\": \"social_framing\", \"min\": -1.0, \"max\": 1.0,
#           \"prompt_negative\": \"frame the topic in terms of systems, institutions, incentives, or abstract structures\",
#           \"prompt_positive\": \"frame the topic in terms of everyday experience, relationships, identity, or interpersonal consequences\"},
#          {\"name\": \"epistemic_style\", \"min\": -1.0, \"max\": 1.0,
#           \"prompt_negative\": \"reason impressionistically, using intuition, analogies, or informal judgment\",
#           \"prompt_positive\": \"reason with explicit uncertainty, evidence standards, caveats, and testable distinctions\"},
#          {\"name\": \"consensus_alignment\", \"min\": -1.0, \"max\": 1.0,
#           \"prompt_negative\": \"challenge prevailing assumptions, surface overlooked counterpoints, or argue against the position most readers likely hold\",
#           \"prompt_positive\": \"build from widely shared premises, emphasize points likely to feel reasonable to most readers, and avoid needlessly contrarian framing\"}
#        ],
#        \"sampling\": {\"method\": \"latin_hypercube\", \"n_agents\": 96, \"seed\": ${pop_seed}},
#        \"evaluator_sampling\": {\"method\": \"latin_hypercube\", \"n_evaluators\": 16, \"seed\": $((1000 + pop_seed))},
#        \"discretization\": {
#          \"enabled\": true,
#          \"bins\": {
#            \"stance_valence\": [-0.5, 0.5],
#            \"social_framing\": [-0.5, 0.5],
#            \"epistemic_style\": [-0.5, 0.5],
#            \"consensus_alignment\": [-0.5, 0.5]
#          }
#        }
#      }
#    }"
#done


#
#for pop_seed in 11 22 33 44 55; do
#  python3 log_collect.py \
#    --reuse_contents_from "runs/run_popseed_${pop_seed}.contents_only.json" \
#    --output "runs/run_popseed_${pop_seed}_reuse_together_evalaudit_compact.json" \
#    --checkpoint_dir "runs/run_popseed_${pop_seed}_reuse_together_evalaudit_compact.checkpoints" \
#    --config_json "{
#      \"post_model\": {
#        \"provider\": \"openai\",
#        \"model\": \"gpt-5.4-mini\"
#      },
#      \"eval_model\": {
#        \"provider\": \"together\",
#        \"model\": \"deepseek-ai/DeepSeek-V3\"
#      },
#      \"audit_model\": {
#        \"provider\": \"together\",
#        \"model\": \"deepseek-ai/DeepSeek-V3\"
#      },
#      \"temperature_post\": 0.7,
#      \"temperature_eval\": 0.1,
#      \"batch_evaluations_across_panel\": true,
#      \"save_full_text\": false,
#      \"save_response_text\": false,
#      \"save_eval_model_metadata\": false,
#      \"save_audit_model_metadata\": false,
#      \"api_max_workers\": 8,
#      \"eval_max_workers\": 8,
#      \"audit_max_workers\": 4,
#      \"api_timeout_sec\": 240,
#      \"api_max_retries\": 3,
#      \"random_seed\": 42,
#      \"n_timesteps\": 200,
#      \"trait_space\": {
#        \"dimensions\": [
#          {
#            \"name\": \"stance_valence\",
#            \"min\": -1.0,
#            \"max\": 1.0,
#            \"prompt_negative\": \"examine the topic from a critical or counter-argument perspective\",
#            \"prompt_positive\": \"develop the strongest reasonable case in favor of the topic's main premise\"
#          },
#          {
#            \"name\": \"social_framing\",
#            \"min\": -1.0,
#            \"max\": 1.0,
#            \"prompt_negative\": \"frame the topic in terms of systems, institutions, incentives, or abstract structures\",
#            \"prompt_positive\": \"frame the topic in terms of everyday experience, relationships, identity, or interpersonal consequences\"
#          },
#          {
#            \"name\": \"epistemic_style\",
#            \"min\": -1.0,
#            \"max\": 1.0,
#            \"prompt_negative\": \"reason impressionistically, using intuition, analogies, or informal judgment\",
#            \"prompt_positive\": \"reason with explicit uncertainty, evidence standards, caveats, and testable distinctions\"
#          },
#          {
#            \"name\": \"consensus_alignment\",
#            \"min\": -1.0,
#            \"max\": 1.0,
#            \"prompt_negative\": \"challenge prevailing assumptions, surface overlooked counterpoints, or argue against the position most readers likely hold\",
#            \"prompt_positive\": \"build from widely shared premises, emphasize points likely to feel reasonable to most readers, and avoid needlessly contrarian framing\"
#          }
#        ],
#        \"sampling\": {
#          \"method\": \"latin_hypercube\",
#          \"n_agents\": 96,
#          \"seed\": ${pop_seed}
#        },
#        \"evaluator_sampling\": {
#          \"method\": \"latin_hypercube\",
#          \"n_evaluators\": 8,
#          \"seed\": $((1000 + pop_seed))
#        },
#        \"discretization\": {
#          \"enabled\": true,
#          \"bins\": {
#            \"stance_valence\": [-0.5, 0.5],
#            \"social_framing\": [-0.5, 0.5],
#            \"epistemic_style\": [-0.5, 0.5],
#            \"consensus_alignment\": [-0.5, 0.5]
#          }
#        }
#      }
#    }"
#done




#for N in 384; do
#  for pop_seed in 11 22 33; do
#    python3 log_collect.py \
#      --output "runs_N${N}/run_popseed_${pop_seed}.json" \
#      --checkpoint_dir "runs_N${N}/run_popseed_${pop_seed}.checkpoints" \
#      --config_json "{
#        \"random_seed\": 42,
#        \"n_timesteps\": 100,
#        \"trait_space\": {
#          \"dimensions\": [
#            {\"name\": \"stance_valence\", \"min\": -1.0, \"max\": 1.0,
#             \"prompt_negative\": \"examine the topic from a critical or counter-argument perspective\",
#             \"prompt_positive\": \"develop the strongest reasonable case in favor of the topic's main premise\"},
#            {\"name\": \"social_framing\", \"min\": -1.0, \"max\": 1.0,
#             \"prompt_negative\": \"frame the topic in terms of systems, institutions, incentives, or abstract structures\",
#             \"prompt_positive\": \"frame the topic in terms of everyday experience, relationships, identity, or interpersonal consequences\"},
#            {\"name\": \"epistemic_style\", \"min\": -1.0, \"max\": 1.0,
#             \"prompt_negative\": \"reason impressionistically, using intuition, analogies, or informal judgment\",
#             \"prompt_positive\": \"reason with explicit uncertainty, evidence standards, caveats, and testable distinctions\"},
#            {\"name\": \"consensus_alignment\", \"min\": -1.0, \"max\": 1.0,
#             \"prompt_negative\": \"challenge prevailing assumptions, surface overlooked counterpoints, or argue against the position most readers likely hold\",
#             \"prompt_positive\": \"build from widely shared premises, emphasize points likely to feel reasonable to most readers, and avoid needlessly contrarian framing\"}
#          ],
#          \"sampling\": {\"method\": \"latin_hypercube\", \"n_agents\": ${N}, \"seed\": ${pop_seed}},
#          \"evaluator_sampling\": {\"method\": \"latin_hypercube\", \"n_evaluators\": 16, \"seed\": $((1000 + pop_seed))},
#          \"discretization\": {
#            \"enabled\": true,
#            \"bins\": {
#              \"stance_valence\": [-0.5, 0.5],
#              \"social_framing\": [-0.5, 0.5],
#              \"epistemic_style\": [-0.5, 0.5],
#              \"consensus_alignment\": [-0.5, 0.5]
#            }
#          }
#        }
#      }"
#  done
#done


