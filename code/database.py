import pickle
import os

DB_DIR = "database"
FACE_DB_PATH = os.path.join(DB_DIR, "face_database.pkl")

# Create the database directory if it doesn't exist
os.makedirs(DB_DIR, exist_ok=True)

# Create an empty dictionary
empty_database = {}

# Save the empty dictionary to the .pkl file
with open(FACE_DB_PATH, 'wb') as f:
    pickle.dump(empty_database, f)

print(f"Success: Empty database created at {FACE_DB_PATH}")