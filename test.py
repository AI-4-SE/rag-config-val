from dotenv import load_dotenv
import os


def main():
    load_dotenv(dotenv_path=".env")

    api_key = os.environ["PINECONE_API_KEY"]
    


if __name__ == "__main__":
    main()