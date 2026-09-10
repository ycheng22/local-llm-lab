import os
import argparse
from datasets import load_dataset
from llm_lab.data.schema import TaskSchema, TestCase
from llm_lab.data.loader import save_dataset

def generate_mbpp_tasks(count: int = 500) -> list[TaskSchema]:
    print("Loading mbpp sanitized dataset...")
    # mbpp sanitized has train (374), validation (90), test (90)
    # We will concatenate them to get enough tasks
    dataset_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    dataset_val = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
    dataset_test = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    
    from datasets import concatenate_datasets
    dataset = concatenate_datasets([dataset_train, dataset_val, dataset_test])
    
    tasks = []
    
    for i, item in enumerate(dataset):
        if i >= count:
            break
            
        task_id = item["task_id"]
        prompt = item["prompt"]
        code = item["code"]
        test_list = item["test_list"]
        
        visible_tests = []
        hidden_tests = []
        
        for idx, test_str in enumerate(test_list):
            if test_str.startswith("assert "):
                test_body = test_str[7:].strip()
                
                if "==" in test_body:
                    parts = test_body.rsplit("==", 1)
                    input_str = parts[0].strip()
                    expected_str = parts[1].strip()
                elif " is " in test_body:
                    parts = test_body.rsplit(" is ", 1)
                    input_str = parts[0].strip()
                    expected_str = parts[1].strip()
                else:
                    input_str = test_body
                    expected_str = "True"
            else:
                input_str = test_str
                expected_str = "True"
                
            tc = TestCase(input=input_str, expected=expected_str)
            if idx == 0:
                visible_tests.append(tc)
            else:
                hidden_tests.append(tc)
                
        starter_code = item.get("test_setup_code", "")
        
        task = TaskSchema(
            id=f"mbpp_{task_id}",
            version=1,
            category="coding",
            difficulty=0.5,
            prompt=f"Write a Python function to solve the following problem:\n{prompt}",
            starter_code=starter_code,
            visible_tests=visible_tests,
            hidden_tests=hidden_tests,
            reference_solution=code,
            metadata={"source": "mbpp_sanitized"}
        )
        tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/train_complex_500.jsonl")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    tasks = generate_mbpp_tasks(args.count)
    save_dataset(tasks, args.output)
    print(f"Generated {len(tasks)} complex tasks to {args.output}")
