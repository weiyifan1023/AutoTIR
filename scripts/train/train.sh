export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL="https://api.bandw.top"  # proxy to wandb


PROMPT_KEY=question
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=8192
APPLY_CHAT=True
PROMPT_TEMPLATE_NAME=autotir_template_sys
ACTOR_MODEL_PATH=/share/project/models/Qwen2.5-7B-Instruct  #/your/model/path
REWARD_MANAGER=re_search
ROLLOUT_NAME=vllm_with_search
ROLLOUT_N=5
ROLLOUT_TP=2
ROLLOUT_GPU_UTIL=0.85
SEARCH_URL="http://127.0.0.1:80" #/your/search/url
SANDBOX_URL="http://127.0.0.1:81"
PROJECT_NAME=  #project-name-on-wandb
EXPERIMENT_NAME= #experiment-name-on-wandb
NNODES=1
N_GPUS_PER_NODE=8
SAVE_FREQ=10
TEST_FREQ=10
TOTAL_EPOCHS=2
WANDB_API_KEY=  # your-wandb-api-key
SWANLAB_API_KEY=  # your-swanlab-api-key

SAVE_PATH=  #/your/save/path
TRAIN_FILES= #/your/train/file/path
TEST_FILES= #/your/test/file/path

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --apply_chat) APPLY_CHAT="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --rollout_tp) ROLLOUT_TP="$2"; shift 2;;
        --rollout_gpu_util) ROLLOUT_GPU_UTIL="$2"; shift 2;;
        --search_url) SEARCH_URL="$2"; shift 2;;
        --sandbox_url) SANDBOX_URL="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --swanlab_api_key) SWANLAB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        --train_files) TRAIN_FILES="$2"; shift 2;;
        --test_files) TEST_FILES="$2"; shift 2;;
        *)
            echo "unknown argument '$1'" >&2
            exit 1;;
    esac
done

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

if [ "$SWANLAB_API_KEY" != "None" ]; then
    swanlab login --relogin $SWANLAB_API_KEY
    export SWANLAB_LOG_DIR=${SAVE_PATH}/swanlab
    export SWANLAB_API_KEY
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.apply_chat=${APPLY_CHAT} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.search_url=${SEARCH_URL} \
    actor_rollout_ref.rollout.sandbox_url=${SANDBOX_URL} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=True \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs | tee ${SAVE_PATH}/run.log