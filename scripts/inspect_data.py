from datasets import load_dataset

HF_DATASET = "tzufi/RVS"

ds = load_dataset(HF_DATASET)
print(ds)

# print splits + columns + one example from the first split
split = list(ds.keys())[0]
print("\nSplit:", split)
print("Columns:", ds[split].column_names)
print("Example[0]:", ds[split][0])
