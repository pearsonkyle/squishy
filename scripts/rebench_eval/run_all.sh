#!/bin/bash
# End-to-end SWE-rebench evaluation pipeline.
#
# Runs inside the llmtk docker container (requires Docker socket mount for
# the eval step — see docker/run.sh).
#
# Usage:
#   # From project root, inside docker:
#   bash scripts/rebench_eval/run_all.sh \
#       --model gemma-4-31b \
#       --base-url http://10.132.212.37:10201/v1 \
#       --tag gemma4-31b-v1 \
#       --sample 5 \
#       --seed 42

set -e

# Defaults
MODEL=""
BASE_URL="http://localhost:1234/v1"
TAG="run-01"
SAMPLE=5
SEED=42
SPLIT="filtered"
MAX_TURNS=50
CONCURRENCY=1
TASK_TIMEOUT=900
EVAL_TIMEOUT=600
SKIP_BENCH=false
SKIP_EVAL=false
PYTHON_VERSION=""
DATASET=""
LANGUAGE=""
COMPAT_FILTER=false
API_KEY="${OPENAI_API_KEY:-local}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --sample) SAMPLE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --compat-filter) COMPAT_FILTER=true; shift ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --task-timeout) TASK_TIMEOUT="$2"; shift 2 ;;
        --eval-timeout) EVAL_TIMEOUT="$2"; shift 2 ;;
        --skip-bench) SKIP_BENCH=true; shift ;;
        --skip-eval) SKIP_EVAL=true; shift ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    exit 1
fi

# v6e: --compat-filter is python-specific (apply_compat_filter only checks
# Python-shaped install configs). Without --language python, multilingual V2
# draws can land Java/JS/Rust/PHP instances that the squishy harness can't
# evaluate (no `pip install -e .` for non-Python). Fail fast so the user
# learns this at submit time, not after a wasted bench run.
if [ "$COMPAT_FILTER" = true ] && [ -z "$LANGUAGE" ]; then
    echo "ERROR: --compat-filter requires --language python (V2 is multilingual)."
    echo "       Add: --language python"
    echo "       (Or remove --compat-filter to opt into the multilingual draw."
    echo "        The squishy harness only supports Python end-to-end today.)"
    exit 1
fi

# Resolve script directory to a path visible to sibling Docker containers.
# Inside the llmtk container /project maps to a host path, but sibling
# containers (swerebench eval images) can't see /project.  The /data/gondor
# and /data/hobbiton mounts are identity-mapped, so prefer those.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == /project/* ]]; then
    # 1) Try readlink (works if /project is a symlink into /data/...).
    REAL_PROJECT=$(readlink -f /project 2>/dev/null || true)
    if [[ "$REAL_PROJECT" == /data/* ]] && [ -d "$REAL_PROJECT" ]; then
        SCRIPT_DIR="${REAL_PROJECT}/${SCRIPT_DIR#/project/}"
    else
        # 2) /project is a real mountpoint (NFS bind). Locate the matching
        #    path under /data/{gondor,hobbiton} by NFS source comparison.
        PROJECT_SRC=$(findmnt -n -o SOURCE /project 2>/dev/null || true)
        if [ -n "$PROJECT_SRC" ]; then
            for ROOT in /data/gondor /data/hobbiton; do
                [ -d "$ROOT" ] || continue
                ROOT_SRC=$(findmnt -n -o SOURCE "$ROOT" 2>/dev/null || true)
                [ -n "$ROOT_SRC" ] || continue
                # If PROJECT_SRC starts with ROOT_SRC, derive subpath.
                if [[ "$PROJECT_SRC" == "$ROOT_SRC"* ]]; then
                    SUBPATH="${PROJECT_SRC#$ROOT_SRC}"
                    CANDIDATE="${ROOT}${SUBPATH}/${SCRIPT_DIR#/project/}"
                    if [ -d "$(dirname "$CANDIDATE")" ] || [ -d "$CANDIDATE" ]; then
                        SCRIPT_DIR="$CANDIDATE"
                        break
                    fi
                fi
            done
        fi
    fi
fi
RESULTS_DIR="${SCRIPT_DIR}/results/${TAG}"

# Fail loudly if eval will be attempted with a /project/* path (sibling
# containers can't see it — eval.sh mount will silently fail).
if [ "$SKIP_EVAL" != true ] && [[ "$RESULTS_DIR" == /project/* ]]; then
    echo "ERROR: RESULTS_DIR=$RESULTS_DIR is under /project/, which sibling Docker"
    echo "       containers cannot see. Eval would silently fail with"
    echo "       'eval.sh: No such file or directory'."
    echo "       Either pass --skip-eval, or run from a host-visible path."
    exit 1
fi

echo "============================================"
echo "  SWE-rebench Evaluation Pipeline"
echo "============================================"
echo "  Model:        ${MODEL}"
echo "  Base URL:     ${BASE_URL}"
echo "  Tag:          ${TAG}"
echo "  Sample:       ${SAMPLE}"
echo "  Seed:         ${SEED}"
echo "  Split:        ${SPLIT}"
echo "  Max turns:    ${MAX_TURNS}"
echo "  Concurrency:  ${CONCURRENCY}"
echo "  Eval timeout: ${EVAL_TIMEOUT}s"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Skip bench:   ${SKIP_BENCH}"
echo "  Skip eval:    ${SKIP_EVAL}"
echo "  Python filter:${PYTHON_VERSION:-<none>}"
echo "  Dataset:      ${DATASET:-<auto-detect>}"
echo "  Language:     ${LANGUAGE:-<none>}"
echo "  Compat filter:${COMPAT_FILTER}"
echo "============================================"
echo

# Step 1: Prepare instances
echo "[Step 1/5] Preparing instances..."
PREP_ARGS=(
    --split "${SPLIT}"
    --sample "${SAMPLE}"
    --seed "${SEED}"
    --tag "${TAG}"
    --output "${RESULTS_DIR}/instances.jsonl"
)
if [ -n "${PYTHON_VERSION}" ]; then
    PREP_ARGS+=(--python-version "${PYTHON_VERSION}")
fi
if [ -n "${DATASET}" ]; then
    PREP_ARGS+=(--dataset "${DATASET}")
fi
if [ -n "${LANGUAGE}" ]; then
    PREP_ARGS+=(--language "${LANGUAGE}")
fi
if [ "$COMPAT_FILTER" = true ]; then
    PREP_ARGS+=(--compat-filter)
fi
python "${SCRIPT_DIR}/prepare_instances.py" "${PREP_ARGS[@]}"
echo

# Step 2: Run squishy-bench
if [ "$SKIP_BENCH" = true ]; then
    echo "[Step 2/5] Skipping squishy-bench (--skip-bench)"
else
    echo "[Step 2/5] Running squishy-bench..."
    squishy-bench swe \
        --instances "${RESULTS_DIR}/instances.jsonl" \
        --output "${RESULTS_DIR}/predictions.jsonl" \
        --workspace-root "${RESULTS_DIR}/workspaces" \
        --model-name "${MODEL}" \
        --model "${MODEL}" \
        --base-url "${BASE_URL}" \
        --api-key "${API_KEY}" \
        --concurrency "${CONCURRENCY}" \
        --task-timeout "${TASK_TIMEOUT}" \
        --max-turns "${MAX_TURNS}" \
        --auto-init
fi
echo

# Step 3: Export training data (all instances)
echo "[Step 3/5] Exporting training data..."
python "${SCRIPT_DIR}/export_training.py" \
    --predictions "${RESULTS_DIR}/predictions.jsonl" \
    --output "${RESULTS_DIR}/training.jsonl"
echo

# Step 4: Evaluate patches using Docker eval images
if [ "$SKIP_EVAL" = true ]; then
    echo "[Step 4/5] Skipping evaluation (--skip-eval)"
else
    echo "[Step 4/5] Evaluating patches..."
    if ! command -v docker &> /dev/null; then
        echo "WARNING: Docker not available — skipping evaluation."
        echo "  Run eval manually on a host with Docker access:"
        echo "  python3 ${SCRIPT_DIR}/run_eval.py \\"
        echo "      --predictions ${RESULTS_DIR}/predictions.jsonl \\"
        echo "      --instances ${RESULTS_DIR}/instances.jsonl \\"
        echo "      --output ${RESULTS_DIR}/eval_results.jsonl"
    else
        python "${SCRIPT_DIR}/run_eval.py" \
            --predictions "${RESULTS_DIR}/predictions.jsonl" \
            --instances "${RESULTS_DIR}/instances.jsonl" \
            --output "${RESULTS_DIR}/eval_results.jsonl" \
            --scratch-dir "${RESULTS_DIR}/.scratch" \
            --timeout "${EVAL_TIMEOUT}"
    fi
fi
echo

# Step 5: Summary
echo "[Step 5/5] Summary"
echo "============================================"
echo "Results in: ${RESULTS_DIR}/"
ls -lh "${RESULTS_DIR}/"
echo
echo "Files:"
echo "  instances.jsonl    - Input instances"
echo "  predictions.jsonl  - Full predictions with transcripts"
echo "  training.jsonl     - SFT training data (all instances)"
if [ -f "${RESULTS_DIR}/eval_results.jsonl" ]; then
    echo "  eval_results.jsonl - Patch evaluation results"
    # Print resolved + partial-credit summary (audit item #6)
    SUMMARY=$(python3 -c "
import json
r=t=0
f2p_passed=f2p_total=0
near_misses=[]
with open('${RESULTS_DIR}/eval_results.jsonl') as f:
    for line in f:
        d=json.loads(line)
        t+=1
        if d.get('resolved'): r+=1
        f2p=d.get('tests',{}).get('fail_to_pass',{})
        p=f2p.get('passed',0); n=f2p.get('total',0)
        f2p_passed+=p; f2p_total+=n
        if 0 < p < n:
            near_misses.append(f\"{d['instance_id']}({p}/{n})\")
out=f'Resolved: {r}/{t}  |  F2P passed: {f2p_passed}/{f2p_total}'
if near_misses:
    out += f'  |  Near-misses: {\", \".join(near_misses)}'
print(out)
" 2>/dev/null || echo "unknown")
    echo "  ${SUMMARY}"
fi
echo
echo "To export only resolved instances for training:"
echo "  python ${SCRIPT_DIR}/export_training.py \\"
echo "      --predictions ${RESULTS_DIR}/predictions.jsonl \\"
echo "      --output ${RESULTS_DIR}/training_resolved.jsonl \\"
echo "      --eval-results ${RESULTS_DIR}/eval_results.jsonl \\"
echo "      --resolved-only"
echo "============================================"
