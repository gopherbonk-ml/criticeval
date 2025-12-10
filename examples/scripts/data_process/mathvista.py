from datasets import load_dataset
from pathlib import Path
from PIL import Image

dataset = load_dataset("AI4Math/MathVista", split="testmini")
dataset = dataset.shuffle(seed=42).select(range(10))


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
 

def process_example(example, idx):
    image = example["decoded_image"]
    image_name = f"image_{idx}.png"

    image.save(str(IMAGES_DIR / image_name))

    problem = example["question"]
    answer = example["answer"]

    return {
        "task": problem,
        "image": image_name,
        "target_answer": answer
    }


dataset = dataset.map(
    process_example,
    with_indices=True,
    remove_columns=dataset.column_names)

dataset.to_csv(OUTPUT_DIR / "math_vista_example.csv")
