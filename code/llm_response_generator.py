import json
import time
import threading
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime, timedelta
from apertus_api import ApertusAPI
import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

class ProgressTracker:
    """Global progress tracker for all APIs"""
    def __init__(self, total_languages=16):
        self.total_languages = total_languages
        self.total_entries = 382 * total_languages  # 382 per language
        self.start_time = time.time()
        self.language_status = {}  # Track by language instead of API
        self.lock = threading.Lock()

    def update(self, language, entries_done, total_entries, turns_done, api_idx):
        with self.lock:
            self.language_status[language] = {
                'entries_done': entries_done,
                'total_entries': total_entries,
                'turns_done': turns_done,
                'api_idx': api_idx,
                'last_update': time.time()
            }
            self.display()

    def display(self):
        os.system('clear' if os.name == 'posix' else 'cls')

        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        print("=" * 80)
        print(f"🚀 PARALLEL LLM RESPONSE GENERATION - REAL-TIME MONITOR")
        print(f"⏱️  Running Time: {elapsed_str}")
        print("=" * 80)

        total_entries_done = 0
        total_turns_done = 0

        # Group by API for display
        api_groups = {0: [], 1: [], 2: [], 3: [], 4: []}

        for lang, status in self.language_status.items():
            api_idx = status.get('api_idx', 0)
            if api_idx in api_groups:
                api_groups[api_idx].append((lang, status))
            total_entries_done += status['entries_done']
            total_turns_done += status['turns_done']

        # Display each API's languages
        for api_idx in range(5):
            langs = api_groups[api_idx]
            if langs:
                print(f"\n[API {api_idx}]")
                for lang, status in langs:
                    progress = (status['entries_done'] / status['total_entries']) * 100 if status['total_entries'] > 0 else 0
                    bar_length = 30
                    filled = int(bar_length * progress / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"  {lang}: [{bar}] {progress:.1f}% ({status['entries_done']}/{status['total_entries']})")
            else:
                print(f"\n[API {api_idx}] No language assigned yet")

        # Overall statistics
        overall_progress = (total_entries_done / self.total_entries) * 100 if self.total_entries > 0 else 0

        print("\n" + "=" * 80)
        print(f"📊 OVERALL PROGRESS")
        print(f"  Total Entries: {total_entries_done}/{self.total_entries} ({overall_progress:.1f}%)")
        print(f"  Total Turns Processed: {total_turns_done}")
        print(f"  Active Languages: {len(self.language_status)}")

        if elapsed > 0:
            rate = total_turns_done / elapsed
            print(f"  Processing Rate: {rate:.1f} turns/sec")

            if rate > 0 and total_entries_done > 0:
                remaining_entries = self.total_entries - total_entries_done
                avg_turns_per_entry = total_turns_done / total_entries_done
                remaining_turns = remaining_entries * avg_turns_per_entry
                eta_seconds = remaining_turns / rate
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                print(f"  Estimated Time Remaining: {eta_str}")

        print("=" * 80)
        print("\nPress Ctrl+C to stop | Logs: logs/llm_realtime.log")
        sys.stdout.flush()

# Global progress tracker
progress_tracker = None

class APIRateLimiter:
    """Shared rate limiter per API key"""
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 0.25  # 4 req/sec
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            self.last_request_time = time.time()

# Global rate limiters for each API key
rate_limiters = {}

def process_language_with_api(api_key: str, api_index: int, language_code: str) -> str:
    """Process a single language with a specific API key"""
    global progress_tracker, rate_limiters

    # Get or create rate limiter for this API
    if api_index not in rate_limiters:
        rate_limiters[api_index] = APIRateLimiter()
    rate_limiter = rate_limiters[api_index]

    api = ApertusAPI([api_key])

    input_file = Path('multilingual_datasets_filtered') / f'mhj_dataset_{language_code}.json'
    output_dir = Path('llm_responses')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'mhj_dataset_{language_code}_with_responses.json'

    # Check if already complete
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if len(existing) >= 382:
                first_entry = existing[0]
                if 'turns' in first_entry and first_entry['turns']:
                    if 'llm_response' in first_entry['turns'][0]:
                        return 'skipped'
        except:
            pass

    if not input_file.exists():
        return 'error'

    # Load dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    results = []
    turns_processed = 0

    # Process each entry
    for i, entry in enumerate(dataset):
        try:
            turns = entry.get('turns', [])
            processed_turns = []
            conversation_history = []

            for turn in turns:
                content = turn.get('content', '')

                if not content:
                    processed_turns.append({
                        **turn,
                        'llm_response': 'ERROR: Empty content',
                        'error': True
                    })
                    continue

                # Wait for rate limit
                rate_limiter.wait_if_needed()

                # Truncate if needed
                if len(content) > 1200:
                    content = content[:1200] + "..."

                messages = conversation_history.copy()
                messages.append({'role': 'user', 'content': content})

                try:
                    # Call API
                    response = api.call_model(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=300
                    )

                    if response:
                        processed_turns.append({
                            **turn,
                            'llm_response': response,
                            'response_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        conversation_history.append({'role': 'user', 'content': content})
                        conversation_history.append({'role': 'assistant', 'content': response})
                        turns_processed += 1
                    else:
                        processed_turns.append({
                            **turn,
                            'llm_response': 'ERROR: No response',
                            'error': True
                        })

                except Exception as e:
                    error_str = str(e)
                    if '429' in error_str:
                        time.sleep(5)
                        # Retry once
                        continue
                    processed_turns.append({
                        **turn,
                        'llm_response': f'ERROR: {error_str[:200]}',
                        'error': True
                    })

            results.append({
                'entry_index': i,
                'turns': processed_turns,
                'api_key_index': api_index,
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    language_code,
                    i + 1,
                    len(dataset),
                    turns_processed,
                    api_index
                )

            # Save intermediate results
            if (i + 1) % 20 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[API {api_index}] Error processing entry {i}: {e}", flush=True)

    # Save final results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 'completed'


class ParallelLLMProcessor:
    """Main processor that coordinates all languages with available APIs"""
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.api_keys = self.config['api_keys'][:5]  # Use first 5 API keys
        self.languages = [
            "kor.Hang", "cmn.Hani", "deu.Latn", "spa.Latn",
            "jpn.Jpan", "fra.Latn", "ita.Latn", "rus.Cyrl",
            "por.Latn", "pol.Latn", "nld.Latn", "ind.Latn",
            "tur.Latn", "ces.Latn", "arb.Arab", "ron.Latn"
        ]

    def process_all(self):
        """Process all languages with dynamic API assignment"""
        global progress_tracker
        progress_tracker = ProgressTracker(len(self.languages))

        results = {}

        # Create thread pool with more workers than APIs to handle all languages
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {}
            api_assignment = {}

            # Submit all languages, cycling through available API keys
            for i, lang_code in enumerate(self.languages):
                api_idx = i % len(self.api_keys)  # Round-robin assignment
                api_key = self.api_keys[api_idx]

                future = executor.submit(
                    process_language_with_api,
                    api_key,
                    api_idx,
                    lang_code
                )
                futures[future] = lang_code
                api_assignment[lang_code] = api_idx
                print(f"[Dispatcher] Assigned {lang_code} to API {api_idx}", flush=True)

            # Collect results as they complete
            for future in as_completed(futures):
                lang_code = futures[future]
                try:
                    result = future.result()
                    results[lang_code] = result
                    print(f"[Dispatcher] {lang_code} finished with status: {result}", flush=True)
                except Exception as e:
                    print(f"[Dispatcher] {lang_code} failed with error: {e}", flush=True)
                    results[lang_code] = 'error'

        # Final summary
        progress_tracker.display()
        print("\n\nPROCESSING COMPLETE!")

        completed = sum(1 for r in results.values() if r == 'completed')
        skipped = sum(1 for r in results.values() if r == 'skipped')
        errors = sum(1 for r in results.values() if r == 'error')

        print(f"Completed: {completed} | Skipped: {skipped} | Errors: {errors}")

        for lang, result in sorted(results.items()):
            api_idx = api_assignment.get(lang, -1)
            print(f"  {lang} (API {api_idx}): {result}")

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Process all 16 languages')
    parser.add_argument('--language', type=str, help='Process single language')

    args = parser.parse_args()

    processor = ParallelLLMProcessor()

    if args.all:
        processor.process_all()
    elif args.language:
        global progress_tracker
        progress_tracker = ProgressTracker(1)
        result = process_language_with_api(
            processor.api_keys[0],
            0,
            args.language
        )
        print(f"\nResult for {args.language}: {result}")
    else:
        print("Usage:")
        print("  python parallel_llm_5api_realtime_fixed.py --all")
        print("  python parallel_llm_5api_realtime_fixed.py --language kor.Hang")


if __name__ == '__main__':
    main()