import json


for name in [
    "W1",
    "b1",
    "W2",
    "b2",
    "W3",
    "b3"
]:

    with open(f"weights/{name}.json") as f:
        data = json.load(f)


    print("\n", name)

    if isinstance(data[0], list):

        print("First row:")
        print(data[0][:5])

    else:

        print("First values:")
        print(data[:5])