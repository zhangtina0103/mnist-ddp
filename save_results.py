import csv

results = [
    ["implementation", "time_s", "accuracy", "epochs", "hardware"],
    ["single GPU",     35,       99.06,       3,        "L40S"],
    ["DDP 2x GPU",     60,       98.94,       3,        "2x L40S"],
]

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results)

print("saved results.csv")
