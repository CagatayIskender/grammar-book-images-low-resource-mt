import os
from run_grammamt import parse_gitdev_igt_file

BASE = r"C:\Users\cagoi\OneDrive\Desktop\Tez\Project\Database\2023glossingST\data"

languages = {
    "Gitksan": ("Gitksan", "git-train-track1-covered.txt", "git-test-track1-uncovered.txt"),
    "Natugu": ("Natugu", "ntu-train-track1-covered", "ntu-test-track1-uncovered"),
    "Lezgi": ("Lezgi", "lez-train-track1-covered", "lez-test-track1-uncovered"),
    "Tsez": ("Tsez", "ddo-train-track1-covered", "ddo-test-track1-uncovered"),
}

print(f"{'Language':<10} | {'Train size':<10} | {'Test size':<10}")
print("-" * 36)

for lang, (folder, train_file, test_file) in languages.items():
    train_path = os.path.join(BASE, folder, train_file)
    test_path = os.path.join(BASE, folder, test_file)

    train_size = len(parse_gitdev_igt_file(train_path))
    test_size = len(parse_gitdev_igt_file(test_path))

    print(f"{lang:<10} | {train_size:<10} | {test_size:<10}")