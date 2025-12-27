import pickle
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class DialSimQA:
    """Question-Answer pair from DialSim dataset"""
    question_id: str  # Unique identifier combining episode, scene, type, category, and index
    questions: Dict[str, str]  # Different versions of the question for different speakers
    options: List[str]  # Multiple choice options
    answer: str  # Correct answer
    category: str  # Question category (e.g., 'past', 'fu', 'ans_w_time', etc.)
    question_type: str  # 'hard' or 'easy'
    episode_name: str  # Episode name
    scene_id: int  # Scene ID

@dataclass
class DialSimScene:
    """A scene from Friends episode in DialSim dataset"""
    scene_id: int
    episode_name: str
    date: str  # Date of the scene
    script: str  # Dialogue script
    hard_questions: List[DialSimQA]
    easy_questions: List[DialSimQA]

@dataclass
class DialSimEpisode:
    """An episode from Friends in DialSim dataset"""
    episode_name: str
    scenes: Dict[int, DialSimScene]

@dataclass
class DialSimSample:
    """A sample from DialSim dataset - represents one episode with all its QA pairs"""
    sample_id: str  # Episode name
    episode: DialSimEpisode
    all_qa: List[DialSimQA]  # All QA pairs from this episode

def parse_dialsim_questions(episode_name: str, scene_id: int, question_type: str,
                            category: str, questions_data: Union[List, Dict]) -> List[DialSimQA]:
    """Parse questions from DialSim dataset.

    Args:
        episode_name: Name of the episode
        scene_id: Scene ID
        question_type: 'hard' or 'easy'
        category: Question category
        questions_data: List of question dicts (for hard_q) or dict of questions (for easy_q)

    Returns:
        List of DialSimQA objects
    """
    qa_list = []

    if question_type == 'hard':
        # Hard questions are in a list
        if questions_data and isinstance(questions_data, list):
            for idx, q_data in enumerate(questions_data):
                if isinstance(q_data, dict) and 'questions' in q_data:
                    question_id = f"{episode_name}_S{scene_id}_{question_type}_{category}_{idx}"
                    qa = DialSimQA(
                        question_id=question_id,
                        questions=q_data['questions'],
                        options=q_data.get('options', []),
                        answer=q_data.get('answer', ''),
                        category=category,
                        question_type=question_type,
                        episode_name=episode_name,
                        scene_id=scene_id
                    )
                    qa_list.append(qa)
    elif question_type == 'easy':
        # Easy questions are in a dict with question_id as key
        if questions_data and isinstance(questions_data, dict):
            for q_id, q_data in questions_data.items():
                if isinstance(q_data, dict) and 'questions' in q_data:
                    question_id = f"{episode_name}_S{scene_id}_{question_type}_{category}_{q_id}"
                    qa = DialSimQA(
                        question_id=question_id,
                        questions=q_data['questions'],
                        options=q_data.get('options', []),
                        answer=q_data.get('answer', ''),
                        category=category,
                        question_type=question_type,
                        episode_name=episode_name,
                        scene_id=scene_id
                    )
                    qa_list.append(qa)

    return qa_list

def parse_dialsim_scene(episode_name: str, scene_id: int, scene_data: dict) -> DialSimScene:
    """Parse a single scene from DialSim dataset.

    Args:
        episode_name: Name of the episode
        scene_id: Scene ID
        scene_data: Scene data dictionary

    Returns:
        DialSimScene object
    """
    hard_questions = []
    easy_questions = []

    # Parse hard questions
    if 'hard_q' in scene_data and scene_data['hard_q']:
        for category, questions in scene_data['hard_q'].items():
            qa_list = parse_dialsim_questions(
                episode_name, scene_id, 'hard', category, questions
            )
            hard_questions.extend(qa_list)

    # Parse easy questions
    if 'easy_q' in scene_data and scene_data['easy_q']:
        for category, questions in scene_data['easy_q'].items():
            qa_list = parse_dialsim_questions(
                episode_name, scene_id, 'easy', category, questions
            )
            easy_questions.extend(qa_list)

    return DialSimScene(
        scene_id=scene_id,
        episode_name=episode_name,
        date=scene_data.get('date', ''),
        script=scene_data.get('script', ''),
        hard_questions=hard_questions,
        easy_questions=easy_questions
    )

def parse_dialsim_episode(episode_name: str, episode_data: dict) -> DialSimEpisode:
    """Parse an episode from DialSim dataset.

    Args:
        episode_name: Name of the episode
        episode_data: Episode data dictionary (scene_id -> scene_data)

    Returns:
        DialSimEpisode object
    """
    scenes = {}

    for scene_id, scene_data in episode_data.items():
        if isinstance(scene_data, dict):
            scene = parse_dialsim_scene(episode_name, scene_id, scene_data)
            scenes[scene_id] = scene

    return DialSimEpisode(
        episode_name=episode_name,
        scenes=scenes
    )

def load_dialsim_dataset(file_path: Union[str, Path],
                         max_episodes: Optional[int] = None,
                         question_categories: Optional[List[str]] = None) -> List[DialSimSample]:
    """Load the DialSim dataset from a pickle file.

    Args:
        file_path: Path to the pickle file containing the dataset
        max_episodes: Maximum number of episodes to load (None = all)
        question_categories: List of question categories to include (None = all)
            Examples: ['past', 'cur', 'fu', 'ans_w_time', 'ans_wo_time', etc.]

    Returns:
        List of DialSimSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    # Load pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from pickle file, got {type(data)}")

    samples = []
    total_qa = 0
    total_hard_qa = 0
    total_easy_qa = 0

    episode_names = list(data.keys())
    if max_episodes:
        episode_names = episode_names[:max_episodes]

    print(f"Loading {len(episode_names)} episodes from DialSim dataset...")

    for episode_idx, episode_name in enumerate(episode_names):
        episode_data = data[episode_name]

        # Parse episode
        episode = parse_dialsim_episode(episode_name, episode_data)

        # Collect all QA pairs from this episode
        all_qa = []
        for scene in episode.scenes.values():
            all_qa.extend(scene.hard_questions)
            all_qa.extend(scene.easy_questions)

        # Filter by category if specified
        if question_categories:
            all_qa = [qa for qa in all_qa if qa.category in question_categories]

        # Count questions
        hard_qa_count = sum(1 for qa in all_qa if qa.question_type == 'hard')
        easy_qa_count = sum(1 for qa in all_qa if qa.question_type == 'easy')

        total_qa += len(all_qa)
        total_hard_qa += hard_qa_count
        total_easy_qa += easy_qa_count

        # Create sample
        sample = DialSimSample(
            sample_id=episode_name,
            episode=episode,
            all_qa=all_qa
        )
        samples.append(sample)

        # Print progress
        if (episode_idx + 1) % 10 == 0:
            print(f"Loaded {episode_idx + 1}/{len(episode_names)} episodes, "
                  f"Total QAs: {total_qa} (Hard: {total_hard_qa}, Easy: {total_easy_qa})")

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total episodes: {len(samples)}")
    print(f"Total QAs: {total_qa}")
    print(f"  Hard questions: {total_hard_qa}")
    print(f"  Easy questions: {total_easy_qa}")
    if samples:
        print(f"Average QAs per episode: {total_qa / len(samples):.2f}")

    return samples

def get_dialsim_statistics(samples: List[DialSimSample]) -> Dict:
    """Get statistics about the DialSim dataset.

    Args:
        samples: List of DialSimSample objects

    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_episodes": len(samples),
        "total_qa_pairs": sum(len(sample.all_qa) for sample in samples),
        "total_scenes": sum(len(sample.episode.scenes) for sample in samples),
        "hard_questions": sum(
            sum(1 for qa in sample.all_qa if qa.question_type == 'hard')
            for sample in samples
        ),
        "easy_questions": sum(
            sum(1 for qa in sample.all_qa if qa.question_type == 'easy')
            for sample in samples
        ),
    }

    # Category distribution
    category_counts = {}
    for sample in samples:
        for qa in sample.all_qa:
            category_counts[qa.category] = category_counts.get(qa.category, 0) + 1

    stats["category_distribution"] = category_counts

    return stats

if __name__ == "__main__":
    # Example usage
    dataset_path = Path(__file__).parent / "data" / "dialsim_v1.1" / "friends_dialsim.pickle"

    try:
        print(f"Loading dataset from: {dataset_path}\n")

        # Load first 3 episodes as a test
        samples = load_dialsim_dataset(dataset_path, max_episodes=3)

        # Get statistics
        stats = get_dialsim_statistics(samples)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if key == "category_distribution":
                print(f"\n{key}:")
                for cat, count in sorted(value.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cat}: {count}")
            else:
                print(f"{key}: {value}")

        # Show a sample question
        if samples and samples[0].all_qa:
            print("\nSample Question:")
            qa = samples[0].all_qa[0]
            print(f"Episode: {qa.episode_name}")
            print(f"Scene: {qa.scene_id}")
            print(f"Type: {qa.question_type}")
            print(f"Category: {qa.category}")
            print(f"Question (default): {qa.questions.get('default', '')}")
            print(f"Options: {qa.options}")
            print(f"Answer: {qa.answer}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
