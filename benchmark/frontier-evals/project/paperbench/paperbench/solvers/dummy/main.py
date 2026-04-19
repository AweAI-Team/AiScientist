import getpass
import os
import sys

if os.environ.get("DEBUG", None) == "1":
    os.system("tail -f /dev/null")  # block indefinitely


# Get the current user's username
username = getpass.getuser()

# Check if the current user ID is 0 (root user ID on Unix-like systems)
if os.getuid() == 0:
    print(f"You are running this script as root. Your username is '{username}'.")
else:
    print(f"You do not have root access. Your username is {username}.")

print("The script is being run with the following python interpreter:")
print(sys.executable)
