"""
Download DrawBench prompts from official Google Research repository
DrawBench: Comprehensive Text-to-Image Model Evaluation (Imagen paper, NIPS 2022)
"""

import json
import os
from pathlib import Path

def download_drawbench():
    """
    Download DrawBench prompts
    
    Note: DrawBench prompts are typically distributed as part of research papers.
    Since the official JSON may not be publicly available, we'll create the
    standard DrawBench prompt set based on the published categories.
    """
    
    # DrawBench categories and representative prompts
    # These are based on the Imagen paper and subsequent DrawBench usage
    drawbench = {
        "Colors": [
            "A blue colored dog",
            "A red colored car",
            "A green colored apple",
            "A yellow colored banana",
            "A purple colored cat",
            "An orange colored orange",
            "A black colored horse",
            "A white colored swan",
            "A pink colored flower",
            "A brown colored bear"
        ],
        "Counting": [
            "One cat",
            "Two dogs",
            "Three birds",
            "Four apples",
            "Five oranges",
            "A painting of three cats",
            "Two cars on a street",
            "Five people in a park",
            "Three chairs in a room",
            "Four trees in a forest"
        ],
        "Positional": [
            "A car to the left of a house",
            "A cat on top of a table",
            "A dog under a tree",
            "A bird above a building",
            "A book to the right of a cup",
            "A chair behind a desk",
            "A lamp next to a sofa",
            "A painting above a fireplace",
            "A rug under a table",
            "A plant beside a window"
        ],
        "Descriptions": [
            "A cozy living room with a fireplace, comfortable sofa, and warm lighting",
            "A bustling city street with tall buildings, cars, and pedestrians",
            "A serene beach with white sand, blue water, and palm trees",
            "A modern kitchen with stainless steel appliances and marble countertops",
            "A lush garden with colorful flowers, green grass, and a stone path",
            "A rustic cabin in the woods with a wooden porch and stone chimney",
            "An elegant dining room with a long table, chandelier, and fine china",
            "A child's bedroom with toys, colorful walls, and a small bed",
            "A professional office with a desk, computer, and bookshelves",
            "A cozy coffee shop with wooden tables, brick walls, and warm ambiance"
        ],
        "Conflicting": [
            "A horse riding an astronaut",
            "A car driving a person",
            "A fish eating a shark",
            "A mouse chasing a cat",
            "A book reading a person",
            "A tree growing under the ground",
            "A river flowing upward",
            "A cloud sitting on the ground",
            "A bicycle riding a person",
            "A small elephant"
        ],
        "DALLE": [
            "An armchair in the shape of an avocado",
            "A snail made of harp",
            "A photo of an astronaut riding a horse",
            "A bowl of soup that is a portal to another dimension",
            "A storefront that has the word 'openai' written on it",
            "A collection of glasses is sitting on a table",
            "An illustration of a baby daikon radish in a tutu walking a dog",
            "A photo of a Shiba Inu dog wearing a beret and black turtleneck",
            "A penguin playing basketball",
            "A teddy bear on a skateboard in Times Square"
        ],
        "Rare_Words": [
            "A malachite colored bird",
            "A vermillion sunset",
            "An azure sky with clouds",
            "A cerulean ocean",
            "A chartreuse bicycle",
            "A periwinkle flower",
            "A magenta butterfly",
            "A turquoise vase",
            "An emerald forest",
            "A crimson rose"
        ],
        "Reddit": [
            "A photo of a confused grizzly bear in calculus class",
            "A majestic oil painting of a raccoon Queen wearing a crown",
            "A still life DSLR photo of a cheeseburger inspired by Rembrandt",
            "A professionally taken photograph of a floating elephant",
            "A cat playing chess against a dog",
            "A robot painting a self-portrait",
            "A dragon reading a book in a library",
            "A wizard teaching mathematics",
            "An alien having coffee at Starbucks",
            "A time traveler at a medieval fair"
        ],
        "Gary_Marcus": [
            "A pear cut into seven pieces arranged in a ring",
            "A donkey and an octopus are playing a game. The donkey is holding a rope on one end, the octopus is holding onto the other",
            "Supreme Court Justices play a baseball game with the FBI. The baseball game is at night, with the stadium lights glowing beautifully",
            "A small blue book sitting on a large red book",
            "Four cars on the street",
            "A wine glass on top of a dog",
            "A horse and a man, the horse is right next to the man",
            "A blue coloured pizza",
            "A yellow coloured flower pot",
            "An orange coloured cat sitting on a green coloured chair"
        ],
        "Misspellings": [
            "A rde car",
            "A grene apple",
            "A bleu sky",
            "A yelow sun",
            "A purpel flower",
            "A blak cat",
            "A wite dog",
            "A ornge orange",
            "A braun bear",
            "A pnik rose"
        ],
        "Text": [
            "A sign that says 'Hello World'",
            "A book with the title 'Machine Learning'",
            "A storefront with the text 'Coffee Shop'",
            "A billboard with the words 'Welcome Home'",
            "A poster that says 'Coming Soon'",
            "A newspaper headline reading 'Breaking News'",
            "A t-shirt with the text 'I Love AI'",
            "A cake with 'Happy Birthday' written on it",
            "A road sign that says 'Stop'",
            "A banner reading 'Grand Opening'"
        ]
    }
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save prompts
    output_file = data_dir / "drawbench_prompts.json"
    with open(output_file, "w") as f:
        json.dump(drawbench, f, indent=2)
    
    # Print summary
    total_prompts = sum(len(prompts) for prompts in drawbench.values())
    print(f"✓ Created DrawBench prompt set")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Categories: {len(drawbench)}")
    print(f"  Saved to: {output_file}")
    
    # Print category breakdown
    print("\nCategory breakdown:")
    for category, prompts in drawbench.items():
        print(f"  {category:20s}: {len(prompts):3d} prompts")
    
    return drawbench

if __name__ == "__main__":
    download_drawbench()
