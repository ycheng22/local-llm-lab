import os
import random
from llm_lab.data import TaskSchema, TestCase, save_dataset

def generate_toy_tasks(count: int = 100) -> list[TaskSchema]:
    tasks = []
    
    # 1. Addition tasks
    for i in range(count // 4):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        task = TaskSchema(
            id=f"toy_add_{i:03d}",
            version=1,
            category="math",
            difficulty=0.1,
            prompt=f"Write a Python function called `add_numbers(x, y)` that returns the sum of x and y.",
            starter_code="def add_numbers(x, y):\n    ",
            visible_tests=[TestCase(input=f"add_numbers({a}, {b})", expected=str(a+b))],
            hidden_tests=[
                TestCase(input="add_numbers(0, 0)", expected="0"),
                TestCase(input="add_numbers(-5, 5)", expected="0"),
                TestCase(input="add_numbers(100, 200)", expected="300")
            ],
            reference_solution="def add_numbers(x, y):\n    return x + y",
            metadata={"source": "synthetic_generator"}
        )
        tasks.append(task)
        
    # 2. String reverse tasks
    for i in range(count // 4):
        word = f"word{i}"
        task = TaskSchema(
            id=f"toy_rev_{i:03d}",
            version=1,
            category="string",
            difficulty=0.2,
            prompt="Write a Python function called `reverse_string(s)` that returns the reversed string.",
            starter_code="def reverse_string(s):\n    ",
            visible_tests=[TestCase(input=f"reverse_string('{word}')", expected=f"'{word[::-1]}'?".replace('?', ''))],
            hidden_tests=[
                TestCase(input="reverse_string('')", expected="''"),
                TestCase(input="reverse_string('a')", expected="'a'"),
                TestCase(input="reverse_string('hello')", expected="'olleh'")
            ],
            reference_solution="def reverse_string(s):\n    return s[::-1]",
            metadata={"source": "synthetic_generator"}
        )
        tasks.append(task)
        
    # 3. List sum tasks
    for i in range(count // 4):
        task = TaskSchema(
            id=f"toy_listsum_{i:03d}",
            version=1,
            category="array",
            difficulty=0.2,
            prompt="Write a Python function called `sum_list(arr)` that returns the sum of a list of numbers.",
            starter_code="def sum_list(arr):\n    ",
            visible_tests=[TestCase(input="sum_list([1, 2, 3])", expected="6")],
            hidden_tests=[
                TestCase(input="sum_list([])", expected="0"),
                TestCase(input="sum_list([-1, 1])", expected="0"),
                TestCase(input="sum_list([10, 20, 30])", expected="60")
            ],
            reference_solution="def sum_list(arr):\n    return sum(arr)",
            metadata={"source": "synthetic_generator"}
        )
        tasks.append(task)
        
    # 4. Palindrome tasks
    for i in range(count - (count // 4) * 3):
        task = TaskSchema(
            id=f"toy_pal_{i:03d}",
            version=1,
            category="string",
            difficulty=0.3,
            prompt="Write a Python function called `is_palindrome(s)` that returns True if the string is a palindrome, False otherwise.",
            starter_code="def is_palindrome(s):\n    ",
            visible_tests=[TestCase(input="is_palindrome('racecar')", expected="True")],
            hidden_tests=[
                TestCase(input="is_palindrome('')", expected="True"),
                TestCase(input="is_palindrome('hello')", expected="False"),
                TestCase(input="is_palindrome('a')", expected="True")
            ],
            reference_solution="def is_palindrome(s):\n    return s == s[::-1]",
            metadata={"source": "synthetic_generator"}
        )
        tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/test.jsonl")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    
    tasks = generate_toy_tasks(args.count)
    save_dataset(tasks, args.output)
    print(f"Generated {len(tasks)} toy tasks to {args.output}")
