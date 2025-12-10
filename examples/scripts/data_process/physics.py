from datasets import load_dataset


dataset = load_dataset("camel-ai/physics", split="train[:2%]")
dataset = dataset.shuffle(seed=42).select(range(10))


def process_example(example):
    topic = example["topic"]
    sub_topic = example["sub_topic"]

    problem = example["message_1"]
    solution = example["message_2"]

    return {
        "source": "camel-ai/physics",
        "topic": f"{topic} | {sub_topic}",
        "difficulty": 5,
        "task": problem,
        "target_solution": solution
    }


dataset = dataset.map(process_example, remove_columns=dataset.column_names)
dataset.to_csv("/workspace/local/repos/criticeval/examples/data/physics_problems.csv")



