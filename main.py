import api
import database_utils
import model_utils
import time

# Huvudfunktionen
def main():
    database_utils.init_db()
    print("Startar...")

    while True:
        sleep_seconds = api.get_map()
        print(f"Väntar {sleep_seconds:.0f} s till nästa request...\n")
        time.sleep(sleep_seconds)
        model_utils.train_and_save()

if __name__ == "__main__":
    main()