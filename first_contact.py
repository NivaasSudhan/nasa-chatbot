import os
import requests
from dotenv import load_dotenv

def get_nasa_apod():
    """
    Connects to the NASA APOD API and fetches today's data.
    """
    # Load environment variables from the .env file
    load_dotenv()

    # Securely get the API key from the environment
    nasa_api_key = os.getenv("NASA_API_KEY")

    if not nasa_api_key:
        print("Error: NASA_API_KEY not found. Make sure it's in your .env file.")
        return

    # The URL for NASA's Astronomy Picture of the Day (APOD) API
    api_url = f"https://api.nasa.gov/planetary/apod?api_key={nasa_api_key}"

    print("Connecting to NASA's APOD API...")

    try:
        # Make the GET request to the API
        response = requests.get(api_url)

        # Raise an exception if the request returned an error (e.g., 404, 500)
        response.raise_for_status()

        # If the request was successful, parse the JSON data
        data = response.json()

        # Print the data in a clean format
        print("\n--- Connection Successful! ---")
        print(f"Title: {data.get('title')}")
        print(f"Date: {data.get('date')}")
        print("\nExplanation:")
        print(data.get('explanation'))

    except requests.exceptions.RequestException as e:
        print(f"\n--- Connection Failed ---")
        print(f"An error occurred: {e}")

# This part ensures the function runs when you execute the script directly
if __name__ == "__main__":
    get_nasa_apod()