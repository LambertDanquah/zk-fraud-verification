import json


files = [
    "weights/W1.json",
    "weights/b1.json",
    "weights/W2.json",
    "weights/b2.json",
    "weights/W3.json",
    "weights/b3.json"
]


for file in files:

    with open(file) as f:
        data = json.load(f)

    print(file)

    if isinstance(data[0], list):
        print(
            "shape:",
            len(data),
            "x",
            len(data[0])
        )
    else:
        print(
            "shape:",
            len(data)
        )

    print()