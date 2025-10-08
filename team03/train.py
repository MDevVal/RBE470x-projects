import subprocess
import sys
import os
import time
import concurrent.futures as cf

def run_variant(variant_num, num_episodes, timeout_sec):
    stats = {
        'win': 0,
        'bomb_death': 0,
        'monster_death': 0,
        'timeout': 0,
        'unknown': 0,
        'error': 0
    }

    for ep in range(1, num_episodes + 1):
        try:
            result = subprocess.run(
                ["python3", f"variant{variant_num}.py"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=os.environ.copy()
            )
            output = (result.stdout or "") + (result.stderr or "")

            print(output)
            if "exit" in output:
                outcome = "win"
            elif "self" in output:
                outcome = "bomb_death"
            elif "monster" in output:
                outcome = "monster_death"
            elif "timeout" in output:
                outcome = "timeout"
            else:
                outcome = "unknown"

        except subprocess.TimeoutExpired:
            outcome = "timeout"
        except Exception:
            outcome = "error"

        stats[outcome] += 1

        total_done = sum(stats.values())
        win_rate = stats['win'] / total_done if total_done else 0
        print(f"[Variant {variant_num}] {ep}/{num_episodes} episodes "
              f"done | win rate {win_rate:.1%}")

    return variant_num, stats


def main():
    num_episodes = 1000
    variants = [1]
    timeout_sec = 30

    missing = [f"variant{v}.py" for v in variants if not os.path.exists(f"variant{v}.py")]
    if missing:
        print("Error: Missing variant files:", ", ".join(missing))
        sys.exit(1)

    print(f"Launching {len(variants)} variants in parallel "
          f"({num_episodes} episodes each)...\n{'='*60}\n")
    start = time.time()

    results = {}
    with cf.ProcessPoolExecutor(max_workers=len(variants)) as executor:
        futures = {
            executor.submit(run_variant, v, num_episodes, timeout_sec): v
            for v in variants
        }

        for fut in cf.as_completed(futures):
            v, stats = fut.result()
            results[v] = stats
            print(f"\n✅ Variant {v} completed!\n{'-'*40}")

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print("ALL VARIANTS COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time / (num_episodes * len(variants)):.2f}s per episode)")
    print(f"{'='*60}\n")

    outcomes = ['win', 'bomb_death', 'monster_death', 'timeout', 'unknown', 'error']
    overall = {k: sum(results[v][k] for v in results) for k in outcomes}
    total_eps = num_episodes * len(variants)

    print("Overall Statistics:")
    for k in outcomes:
        print(f"  {k:15s}: {overall[k]:5d} / {total_eps} ({overall[k]/total_eps:.1%})")

    print("\nPer-Variant Statistics:")
    for v in variants:
        s = results[v]
        print(f"\n  Variant {v}:")
        for k in outcomes:
            print(f"    {k:15s}: {s[k]:5d} / {num_episodes} ({s[k]/num_episodes:.1%})")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

