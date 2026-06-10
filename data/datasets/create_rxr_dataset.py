#!/usr/bin/env python3
"""
Script to randomly select 100 English episodes from RxR dataset and save in R2R format.

This script:
1. Loads the RxR dataset (val_unseen_guide.json.gz and val_unseen_guide_gt.json.gz)
2. Filters for English-only episodes (en-US or en-IN)
3. Randomly selects 100 episodes
4. Converts them to R2R format
5. Saves to datasets/datasets/RxR_VLNCE_v0/val_unseen/
"""

import json
import random
import gzip
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "datasets", "datasets")


def load_json(filepath):
    """Load JSON or JSON.gz file."""
    print(f"Loading {filepath}...")
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def save_json_gz(data, filepath):
    """Save data to compressed JSON.gz file."""
    print(f"Saving to {filepath}...")
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filepath}")


def filter_english_episodes(rxr_data):
    """Filter episodes that have English language (en-US or en-IN)."""
    english_episodes = []
    episodes = rxr_data["episodes"]

    for idx, episode in enumerate(episodes):
        language = episode["instruction"].get("language", "")
        if language in ["en-US", "en-IN"]:
            english_episodes.append((idx, episode))

    return english_episodes


def convert_rxr_to_r2r_format(rxr_episode, rxr_gt_episode, episode_id):
    """Convert RxR episode format to R2R format."""
    reference_path = rxr_episode["reference_path"]
    geodesic_distance = 0.0
    for i in range(len(reference_path) - 1):
        p1 = reference_path[i]
        p2 = reference_path[i + 1]
        dist = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
        geodesic_distance += dist

    r2r_episode = {
        "episode_id": episode_id,
        "goals": rxr_episode["goals"],
        "info": {"geodesic_distance": geodesic_distance},
        "instruction": {
            "instruction_text": rxr_episode["instruction"]["instruction_text"],
            "instruction_tokens": rxr_episode["instruction"].get("instruction_tokens", []),
        },
        "reference_path": rxr_episode["reference_path"],
        "scene_id": rxr_episode["scene_id"],
        "start_position": rxr_episode["start_position"],
        "start_rotation": rxr_episode["start_rotation"],
        "trajectory_id": rxr_episode["trajectory_id"],
    }

    return r2r_episode


def select_random_episodes(episodes_with_distances, num_episodes=100):
    """Randomly select episodes, ensuring unique trajectory IDs."""
    print(f"\nTotal available English episodes: {len(episodes_with_distances)}")

    trajectory_groups = {}
    for (idx, episode), distance in episodes_with_distances:
        traj_id = episode["trajectory_id"]
        if traj_id not in trajectory_groups:
            trajectory_groups[traj_id] = []
        trajectory_groups[traj_id].append(((idx, episode), distance))

    print(f"Unique trajectory IDs: {len(trajectory_groups)}")

    random.seed(42)
    unique_episodes = []
    for traj_id, episodes in trajectory_groups.items():
        chosen = random.choice(episodes)
        unique_episodes.append(chosen)

    if len(unique_episodes) < num_episodes:
        print(
            f"  WARNING: Only {len(unique_episodes)} unique trajectories available, wanted {num_episodes}"
        )
        num_episodes = len(unique_episodes)

    selected = random.sample(unique_episodes, num_episodes)
    print(f"Randomly selected {len(selected)} episodes with unique trajectory IDs")

    return [ep for ep, _ in selected]


def main():
    # Input paths
    rxr_guide_path = os.path.join(
        DATASETS_DIR, "RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz"
    )
    rxr_gt_path = os.path.join(
        DATASETS_DIR, "RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json.gz"
    )

    # R2R instruction vocab source
    r2r_path = os.path.join(
        DATASETS_DIR,
        "R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz",
    )

    # Output path
    output_dir = os.path.join(DATASETS_DIR, "RxR_VLNCE_v0/val_unseen")
    os.makedirs(output_dir, exist_ok=True)

    output_episodes_path = os.path.join(output_dir, "OpenNav_RXR-CE_100_bertidx.json.gz")
    output_gt_path = os.path.join(output_dir, "OpenNav_RXR-CE_100_bertidx_gt.json.gz")

    # Load RxR datasets
    rxr_guide = load_json(rxr_guide_path)
    rxr_gt = load_json(rxr_gt_path)

    # Filter English episodes
    print("\nFiltering English episodes (en-US and en-IN)...")
    english_episodes = filter_english_episodes(rxr_guide)
    print(f"Found {len(english_episodes)} English episodes")

    # Calculate geodesic distance for each episode
    print("\nCalculating geodesic distances...")
    episodes_with_distances = []
    for idx, episode in english_episodes:
        reference_path = episode["reference_path"]
        geodesic_distance = 0.0
        for i in range(len(reference_path) - 1):
            p1 = reference_path[i]
            p2 = reference_path[i + 1]
            dist = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            geodesic_distance += dist
        episodes_with_distances.append(((idx, episode), geodesic_distance))

    # Randomly select 100 episodes
    selected = select_random_episodes(episodes_with_distances, num_episodes=100)

    # Load R2R instruction vocab
    print("\nLoading R2R instruction vocab...")
    r2r_data = load_json(r2r_path)
    instruction_vocab = r2r_data["instruction_vocab"]

    # Convert to R2R format
    print("\nConverting to R2R format...")
    r2r_episodes = []
    r2r_gt_dict = {}

    for new_id, (original_idx, rxr_episode) in enumerate(selected):
        rxr_episode_id = rxr_episode["episode_id"]

        if str(rxr_episode_id) in rxr_gt:
            rxr_gt_episode = rxr_gt[str(rxr_episode_id)]
            r2r_gt_dict[str(new_id)] = {
                "actions": rxr_gt_episode["actions"],
                "locations": rxr_gt_episode["locations"],
                "forward_steps": rxr_gt_episode["forward_steps"],
            }
        else:
            print(f"WARNING: No GT data found for episode {rxr_episode_id}")

        r2r_episode = convert_rxr_to_r2r_format(rxr_episode, rxr_gt_episode, new_id)
        r2r_episodes.append(r2r_episode)

        if (new_id + 1) % 10 == 0:
            print(f"Converted {new_id + 1}/{len(selected)} episodes")

    # Create final dataset structure
    r2r_dataset = {
        "episodes": r2r_episodes,
        "instruction_vocab": instruction_vocab,
    }

    # Save
    save_json_gz(r2r_dataset, output_episodes_path)
    save_json_gz(r2r_gt_dict, output_gt_path)

    # Statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total RxR episodes: {len(rxr_guide['episodes'])}")
    print(f"English episodes: {len(english_episodes)}")
    print(f"Selected episodes: {len(selected)}")
    print(f"\nOutput files:")
    print(f"  Episodes: {output_episodes_path}")
    print(f"  GT:       {output_gt_path}")

    distances = [ep["info"]["geodesic_distance"] for ep in r2r_episodes]
    print(f"\nDistance statistics:")
    print(f"  Mean:   {sum(distances) / len(distances):.2f}m")
    print(f"  Min:    {min(distances):.2f}m")
    print(f"  Max:    {max(distances):.2f}m")

    print(f"\nLanguage distribution in selected episodes:")
    lang_count = {}
    for _, (_, ep) in enumerate(selected):
        lang = ep["instruction"]["language"]
        lang_count[lang] = lang_count.get(lang, 0) + 1
    for lang, count in sorted(lang_count.items()):
        print(f"  {lang}: {count} episodes")
    print("=" * 50)


if __name__ == "__main__":
    main()
