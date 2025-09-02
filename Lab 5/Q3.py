import hashlib
import random
import string
import time

# -------------------------
# Generate random strings
# -------------------------
def generate_random_strings(n, length=20):
    dataset = []
    for _ in range(n):
        rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        dataset.append(rand_str)
    return dataset

# -------------------------
# Hashing functions
# -------------------------
def compute_hash(data, algorithm):
    if algorithm == 'md5':
        return hashlib.md5(data.encode()).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(data.encode()).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(data.encode()).hexdigest()
    else:
        raise ValueError("Unsupported algorithm")

# -------------------------
# Collision detection
# -------------------------
def detect_collisions(hashes):
    seen = {}
    collisions = []
    for original, h in hashes.items():
        if h in seen:
            collisions.append((original, seen[h]))
        else:
            seen[h] = original
    return collisions

# -------------------------
# Experiment
# -------------------------
def run_experiment(num_strings):
    dataset = generate_random_strings(num_strings)

    algorithms = ['md5', 'sha1', 'sha256']
    results = {}

    for algo in algorithms:
        start_time = time.perf_counter()
        hash_map = {s: compute_hash(s, algo) for s in dataset}
        elapsed_time = time.perf_counter() - start_time

        collisions = detect_collisions(hash_map)

        results[algo] = {
            'time': elapsed_time,
            'collisions': collisions,
            'collision_count': len(collisions)
        }

    return results

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    num_strings = random.randint(50, 100)
    print(f"Generating {num_strings} random strings...\n")

    results = run_experiment(num_strings)

    for algo, data in results.items():
        print(f"Algorithm: {algo.upper()}")
        print(f"  Time taken: {data['time']:.6f} seconds")
        print(f"  Collisions found: {data['collision_count']}")
        if data['collision_count'] > 0:
            print(f"  Collision pairs: {data['collisions']}")
        print("-" * 40)
