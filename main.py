import sys

from Pipelines import main_cnn
from Pipelines import main_rf
from Pipelines import main_svm

MODELS = {
    "1": ("CNN (MediaPipe + TFLite)", main_cnn.run),
    "2": ("Random Forest", main_rf.run),
    "3": ("SVM", main_svm.run),
}


def print_menu():
    print("\n===== ASL Recognition - Model Selection =====")
    for key, (name, _) in MODELS.items():
        print(f"  [{key}] {name}")
    print("  [Q] Exit Program")
    print("==============================================")


def main():
    while True:
        print_menu()
        choice = input("Please select an option: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Exiting program.")
            break

        if choice not in MODELS:
            print("Invalid option, please try again.")
            continue

        name, run_pipeline = MODELS[choice]
        print(f"\nLaunching model: {name}")
        print("(In camera window: Press [ESC] to return to menu, [Q] to exit)\n")

        try:
            action = run_pipeline()
        except KeyboardInterrupt:
            print("\nInterrupt detected. Returning to menu.")
            action = "menu"
        except Exception as exc:
            print(f"[Error] An exception occurred while running {name}: {exc}")
            action = "menu"

        if action == "quit":
            print("Program exited.")
            break
        # action == "menu" -> Returns to while loop and displays the menu again


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram exited.")
        sys.exit(0)