from memory_layer import LLMController, AgenticMemorySystem
import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from load_dialsim_dataset import load_dialsim_dataset, DialSimQA, DialSimSample
import nltk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from utils import calculate_metrics, aggregate_metrics
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

class advancedMemAgent:
    def __init__(self, model, backend, retrieve_k, temperature, sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature = temperature

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def retrieve_memory_llm(self, memories_text, query):
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text.
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""

            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "relevant_parts": {
                                        "type": "string",
                                    }
                                },
                                "required": ["relevant_parts"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        return response

    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text.

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""

            # Get LLM response
        response = self.retriever_llm.llm.get_completion(prompt,response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def answer_question(self, qa: DialSimQA) -> str:
        """Generate answer for a DialSim question given the conversation context.

        Args:
            qa: DialSimQA object containing question and options

        Returns:
            Tuple of (prediction, user_prompt, raw_context)
        """
        # Use default question format
        question = qa.questions.get('default', list(qa.questions.values())[0])

        # Generate keywords from question
        keywords = self.generate_query_llm(question)

        # Retrieve relevant memories
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)
        context = raw_context

        # Format options for multiple choice
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(qa.options)])

        # Create prompt based on question type
        user_prompt = f"""Based on the context: {context}

Question: {question}

Available options:
{options_text}

Select the most appropriate answer from the options above. Respond with the exact text of the option you choose."""

        # Get response from LLM
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,
            response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }},
            temperature=self.temperature
        )

        return response, user_prompt, raw_context

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('dialsim_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def evaluate_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    max_episodes: Optional[int] = None,
    max_questions_per_episode: Optional[int] = None,
    question_categories: Optional[List[str]] = None,
    backend: str = "sglang",
    temperature: float = 0.7,
    retrieve_k: int = 10,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000
):
    """Evaluate the agent on the DialSim dataset.

    Args:
        dataset_path: Path to the dataset pickle file
        model: Name of the model to use
        output_path: Path to save results
        max_episodes: Maximum number of episodes to evaluate (None = all)
        max_questions_per_episode: Maximum questions per episode (None = all)
        question_categories: List of question categories to include (None = all)
        backend: Backend to use (openai, ollama, or sglang)
        temperature: Temperature for the model
        retrieve_k: Number of memories to retrieve
        sglang_host: SGLang server host
        sglang_port: SGLang server port
    """
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    categories_str = "_".join(question_categories) if question_categories else "all"
    log_filename = f"eval_dialsim_{model}_{backend}_ep{max_episodes}_cat{categories_str}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")

    # Load dataset
    samples = load_dialsim_dataset(
        dataset_path,
        max_episodes=max_episodes,
        question_categories=question_categories
    )
    logger.info(f"Loaded {len(samples)} episodes")

    # Store results
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)

    # Create memory cache directory
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        f"cached_memories_dialsim_{backend}_{model}"
    )
    os.makedirs(memories_dir, exist_ok=True)

    error_num = 0

    # Evaluate each episode
    for sample_idx, sample in enumerate(samples):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing episode {sample_idx + 1}/{len(samples)}: {sample.sample_id}")
        logger.info(f"{'='*60}")

        # Initialize agent for this episode
        agent = advancedMemAgent(model, backend, retrieve_k, temperature, sglang_host, sglang_port)

        # Create memory cache filename
        safe_episode_name = sample.sample_id.replace('.txt', '').replace(' ', '_')
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_{safe_episode_name}.pkl"
        )
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_{safe_episode_name}.pkl"
        )
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_embeddings_{safe_episode_name}.npy"
        )

        # Check if cached memories exist
        if os.path.exists(memory_cache_file):
            logger.info(f"Loading cached memories for episode {sample.sample_id}")
            try:
                with open(memory_cache_file, 'rb') as f:
                    cached_memories = pickle.load(f)

                # Restore memories to agent
                agent.memory_system.memories = cached_memories

                if os.path.exists(retriever_cache_file):
                    logger.info(f"Loading cached retriever")
                    agent.memory_system.retriever = agent.memory_system.retriever.load(
                        retriever_cache_file, retriever_cache_embeddings_file
                    )
                else:
                    logger.info(f"No retriever cache found, loading from memory")
                    agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                        cached_memories, 'all-MiniLM-L6-v2'
                    )

                logger.info(f"Successfully loaded {len(cached_memories)} memories")
            except Exception as e:
                logger.info(f"Error loading cached memories: {e}. Will recreate memories.")
                cached_memories = None
        else:
            logger.info(f"No cached memories found for episode {sample.sample_id}. Creating new memories.")
            cached_memories = None

            # Add all scenes to memory
            for scene_id, scene in sorted(sample.episode.scenes.items()):
                # Parse dialogue from script and add to memory
                script_lines = scene.script.strip().split('\n')
                for line in script_lines:
                    line = line.strip()
                    if line and ':' in line:
                        # Parse speaker and text
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            speaker = parts[0].strip()
                            text = parts[1].strip()
                            conversation_tmp = f"{speaker} says: {text}"
                            agent.add_memory(conversation_tmp, time=scene.date)

            # Cache memories
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)

            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
            logger.info(f"Successfully cached {len(memories_to_cache)} memories")

        # Get questions for this episode
        questions = sample.all_qa
        if max_questions_per_episode and len(questions) > max_questions_per_episode:
            questions = random.sample(questions, max_questions_per_episode)
            logger.info(f"Sampled {max_questions_per_episode} questions from {len(sample.all_qa)} total")

        # Process each question
        for qa_idx, qa in enumerate(questions):
            total_questions += 1
            category_counts[qa.category] += 1

            # Generate prediction
            prediction, user_prompt, raw_context = agent.answer_question(qa)
            try:
                prediction = json.loads(prediction)["answer"]
            except:
                prediction = prediction.strip()
                logger.info(f"Failed to parse prediction as JSON: {prediction}")
                error_num += 1

            # Log results
            logger.info(f"\nQuestion {total_questions} (Episode {sample_idx + 1}, Q{qa_idx + 1}/{len(questions)})")
            logger.info(f"Category: {qa.category} ({qa.question_type})")
            logger.info(f"Question: {qa.questions.get('default', '')}")
            logger.info(f"Options: {qa.options}")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Reference: {qa.answer}")
            logger.info(f"User Prompt: {user_prompt}")
            logger.info(f"Raw Context: {raw_context}")

            # Calculate metrics (exact match for multiple choice)
            metrics = calculate_metrics(prediction, qa.answer) if qa.answer else {
                "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
            }

            all_metrics.append(metrics)
            all_categories.append(qa.category)

            # Store individual result
            result = {
                "episode": sample.sample_id,
                "question_id": qa.question_id,
                "question": qa.questions.get('default', ''),
                "prediction": prediction,
                "reference": qa.answer,
                "category": qa.category,
                "question_type": qa.question_type,
                "options": qa.options,
                "metrics": metrics
            }
            results.append(result)

            # Log progress
            if total_questions % 50 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Progress: Processed {total_questions} questions")
                logger.info(f"{'='*60}")

    # Calculate aggregate metrics
    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    # Prepare final results
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "total_episodes": len(samples),
        "max_questions_per_episode": max_questions_per_episode,
        "question_categories": question_categories,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    logger.info(f"Error number: {error_num}")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Log summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info(f"Total episodes: {len(samples)}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {category}: {count} questions ({count/total_questions*100:.1f}%)")

    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")

    return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on DialSim dataset")
    parser.add_argument("--dataset", type=str,
                        default="data/dialsim_v1.1/friends_dialsim.pickle",
                        help="Path to the dataset pickle file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to evaluate (default: all)")
    parser.add_argument("--max_questions_per_episode", type=int, default=None,
                        help="Maximum questions per episode (default: all)")
    parser.add_argument("--question_categories", type=str, nargs='+', default=None,
                        help="Question categories to include (e.g., past cur fu ans_w_time)")
    parser.add_argument("--backend", type=str, default="sglang",
                        help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")

    args = parser.parse_args()

    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None

    evaluate_dataset(
        dataset_path=dataset_path,
        model=args.model,
        output_path=output_path,
        max_episodes=args.max_episodes,
        max_questions_per_episode=args.max_questions_per_episode,
        question_categories=args.question_categories,
        backend=args.backend,
        temperature=args.temperature,
        retrieve_k=args.retrieve_k,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port
    )

if __name__ == "__main__":
    main()
