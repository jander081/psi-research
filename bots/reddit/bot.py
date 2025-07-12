# bots/reddit/bot.py

import yaml
import praw
from transformers import pipeline

# Load your config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

reddit_config = config["reddit"]

# Create Reddit instance
reddit = praw.Reddit(
    client_id=reddit_config["client_id"],
    client_secret=reddit_config["client_secret"],
    username=reddit_config["username"],
    password=reddit_config["password"],
    user_agent=reddit_config["user_agent"]
)

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def main():
    # Choose a subreddit to test
    # subreddit_name = "UFOs"
    subreddit_name = "test"

    subreddit = reddit.subreddit(subreddit_name)

    print(f"\nFetching top 3 hot posts from r/{subreddit_name}...\n")

    for submission in subreddit.hot(limit=3):
        print(f"Title: {submission.title}")
        print(f"URL: {submission.url}")

        # Get the submission text (selftext)
        if submission.selftext:
            text = submission.selftext
            # Hugging Face models have input length limits (~1024 tokens)
            if len(text) > 1000:
                text = text[:1000]

            summary = summarizer(text, max_length=50, 
                                 min_length=10, do_sample=False)[0]['summary_text']
            print(f"Summary: {summary}")

            # Footer for transparency
            comment_body = (
            f"**Automated Summary:**\n\n{summary}\n\n"
            "*I am an experimental research bot developed for the PSI Research Project. "
            "If you have feedback or want to report an issue, reply below.*"
            )
            # Post the summary as a comment and log the comment ID
            result = submission.reply(comment_body)
            print("✅ Comment posted.\n")

            # Log the comment ID, submission ID, and title
            with open("comments_log.txt", "a") as log_file:
                log_file.write(f"{result.id} | {submission.id} | {submission.title}\n")

        else:
            print("No text to summarize.")

        print("-" * 40)

if __name__ == "__main__":
    main()