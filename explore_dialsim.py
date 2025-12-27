import pickle

# Load the dialsim dataset
with open(r'e:\1-ntu\dissertation\论文代码\A-mem\data\dialsim_v1.1\friends_dialsim.pickle', 'rb') as f:
    data = pickle.load(f)

print(f"Total episodes: {len(data)}")
print(f"Episode keys (first 5): {list(data.keys())[:5]}\n")

# Explore first episode
first_ep_key = list(data.keys())[0]
first_episode = data[first_ep_key]
print(f"Episode: {first_ep_key}")
print(f"Number of scenes: {len(first_episode)}")
print(f"Scene IDs: {list(first_episode.keys())[:10]}\n")

# Explore first scene
first_scene_id = list(first_episode.keys())[0]
first_scene = first_episode[first_scene_id]
print(f"Scene {first_scene_id}:")
print(f"Keys: {list(first_scene.keys())}")
print(f"Date: {first_scene.get('date')}")
print(f"Script preview: {first_scene.get('script')[:200]}\n")

# Check questions
print("=" * 50)
print("Looking for questions in dataset...")
print("=" * 50)

total_hard_q = 0
total_easy_q = 0
sample_questions = []

for ep_idx, (ep_key, episode) in enumerate(data.items()):
    if ep_idx >= 3:  # Only check first 3 episodes
        break

    for scene_id, scene in episode.items():
        # Check hard questions
        if 'hard_q' in scene:
            for cat, qs in scene['hard_q'].items():
                if qs and len(qs) > 0:
                    total_hard_q += len(qs)
                    if len(sample_questions) < 5:
                        sample_questions.append({
                            'episode': ep_key,
                            'scene_id': scene_id,
                            'type': 'hard',
                            'category': cat,
                            'question': qs[0]
                        })

        # Check easy questions
        if 'easy_q' in scene:
            for cat, qs_dict in scene['easy_q'].items():
                if qs_dict and isinstance(qs_dict, dict):
                    total_easy_q += len(qs_dict)
                    if len(sample_questions) < 10:
                        first_q_key = list(qs_dict.keys())[0] if qs_dict else None
                        if first_q_key:
                            sample_questions.append({
                                'episode': ep_key,
                                'scene_id': scene_id,
                                'type': 'easy',
                                'category': cat,
                                'question': first_q_key,
                                'answer': qs_dict[first_q_key]
                            })

print(f"\nTotal hard questions (first 3 episodes): {total_hard_q}")
print(f"Total easy questions (first 3 episodes): {total_easy_q}")
print(f"\nSample questions:")
for i, sq in enumerate(sample_questions[:5]):
    print(f"\n{i+1}. Episode: {sq['episode']}, Scene: {sq['scene_id']}")
    print(f"   Type: {sq['type']}, Category: {sq['category']}")
    if sq['type'] == 'easy':
        print(f"   Question: {sq['question']}")
        print(f"   Answer: {sq['answer']}")
    else:
        print(f"   Question: {sq['question']}")
