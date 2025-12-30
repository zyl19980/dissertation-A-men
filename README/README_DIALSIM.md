# DialSim Dataset Evaluation

This guide explains how to run A-mem evaluation on the DialSim dataset.

## Files Created

1. **load_dialsim_dataset.py** - Dataset loader for DialSim pickle files
2. **test_advanced_dialsim.py** - Main evaluation script for DialSim dataset
3. **explore_dialsim.py** - Helper script to explore dataset structure

## Dataset Structure

The DialSim dataset contains Friends TV show episodes with:
- **Episodes**: 118 episodes from Friends TV show
- **Scenes**: Each episode has multiple scenes with dates and dialogue scripts
- **Questions**: Two types of questions per scene:
  - **Hard questions** (14,317 in first 3 episodes): Future reasoning questions like 'past', 'cur', 'fu', 'past_past', 'fu_fu', etc.
  - **Easy questions** (516 in first 3 episodes): Time-based and answerable questions like 'ans_w_time', 'ans_wo_time', etc.

## Usage Examples

### 1. Test with first 3 episodes, all questions
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model gpt-4o-mini \
    --backend openai \
    --max_episodes 3 \
    --retrieve_k 10 \
    --temperature 0.7
```

### 2. Test with specific question categories
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model gpt-4o-mini \
    --backend openai \
    --max_episodes 5 \
    --question_categories ans_w_time ans_wo_time past cur fu \
    --retrieve_k 10
```

### 3. Test with limited questions per episode
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model gpt-4o-mini \
    --backend openai \
    --max_episodes 10 \
    --max_questions_per_episode 50 \
    --retrieve_k 10
```

### 4. Using SGLang backend
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --backend sglang \
    --sglang_host http://localhost \
    --sglang_port 30000 \
    --max_episodes 3 \
    --retrieve_k 10
```

### 5. Using Ollama backend
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model llama2 \
    --backend ollama \
    --max_episodes 3 \
    --retrieve_k 10
```

## Command Line Arguments

- `--dataset`: Path to the DialSim pickle file (default: data/dialsim_v1.1/friends_dialsim.pickle)
- `--model`: Model name to use (e.g., gpt-4o-mini, llama2)
- `--backend`: Backend to use (openai, ollama, or sglang)
- `--output`: Path to save evaluation results JSON file
- `--max_episodes`: Maximum number of episodes to evaluate (None = all 118 episodes)
- `--max_questions_per_episode`: Maximum questions per episode (None = all questions)
- `--question_categories`: Space-separated list of categories to include
- `--retrieve_k`: Number of memories to retrieve for each question (default: 10)
- `--temperature`: Sampling temperature for the model (default: 0.7)
- `--sglang_host`: SGLang server host (default: http://localhost)
- `--sglang_port`: SGLang server port (default: 30000)

## Available Question Categories

### Hard Questions
- `past`: Past events
- `cur`: Current events
- `fu`: Future events
- `past_past`: Past-to-past reasoning
- `past_cur`: Past-to-current reasoning
- `past_fu`: Past-to-future reasoning
- `cur_past`: Current-to-past reasoning
- `cur_cur`: Current-to-current reasoning
- `cur_fu`: Current-to-future reasoning
- `fu_past`: Future-to-past reasoning
- `fu_cur`: Future-to-current reasoning
- `fu_fu`: Future-to-future reasoning

### Easy Questions
- `ans_w_time`: Answerable questions with time context
- `ans_wo_time`: Answerable questions without time context
- `before_event_unans`: Unanswerable questions about events before they occur
- `dont_know_unans`: Unanswerable questions
- `dont_know_unans_time`: Unanswerable questions with time context

## Memory Caching

The script automatically caches memories for each episode to speed up repeated evaluations:
- Cached files are stored in `cached_memories_dialsim_{backend}_{model}/`
- Memory cache: `memory_cache_{episode_name}.pkl`
- Retriever cache: `retriever_cache_{episode_name}.pkl` and `retriever_cache_embeddings_{episode_name}.npy`

To force regeneration of memories, delete the cache directory.

## Logs

Evaluation logs are automatically saved to `logs/` directory with timestamp:
- Format: `eval_dialsim_{model}_{backend}_ep{max_episodes}_cat{categories}_{timestamp}.log`

## Output

Results are saved as JSON with:
- Model and dataset information
- Total questions and episodes evaluated
- Category distribution
- Aggregate metrics (F1, ROUGE, BLEU, etc.)
- Individual question results

## Differences from LoComo Dataset

1. **Question Format**: DialSim uses multiple-choice questions with options
2. **Question Types**: Different category system (past/cur/fu vs. categories 1-5)
3. **Script Format**: Dialogue is in script format with speaker names
4. **Structure**: Organized by episodes and scenes rather than conversations and sessions

## Example Output

```
Loading 3 episodes from DialSim dataset...
Overall Statistics:
Total episodes: 3
Total QAs: 14833
  Hard questions: 14317
  Easy questions: 516
Average QAs per episode: 4944.33
```
