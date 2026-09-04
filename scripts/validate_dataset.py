import argparse
import sys
from collections import Counter
from llm_lab.data import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Validate a dataset JSONL file")
    parser.add_argument("file", type=str, help="Path to the JSONL dataset")
    args = parser.parse_args()

    print(f"Validating dataset {args.file}...")
    try:
        tasks = load_dataset(args.file)
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    errors = []
    
    # Check duplicate IDs
    ids = [t.id for t in tasks]
    id_counts = Counter(ids)
    duplicates = [i for i, count in id_counts.items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate task IDs found: {duplicates}")
        
    # Check difficulty values
    for task in tasks:
        if not (0.0 <= task.difficulty <= 1.0):
            errors.append(f"Task {task.id} has invalid difficulty {task.difficulty} (must be 0-1)")
            
        if not task.hidden_tests:
            errors.append(f"Task {task.id} has no hidden tests")

    if errors:
        print("\nValidation Failed. Errors found:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)
        
    print(f"Validation successful! {len(tasks)} tasks loaded and verified.")

if __name__ == "__main__":
    main()
