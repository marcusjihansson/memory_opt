import dspy


# Define a simple signature
# Signatures describe the input/output behavior of a module
class Sentiment(dspy.Signature):
    """Analyze the sentiment of a text."""

    text = dspy.InputField(desc="The text to analyze")
    sentiment = dspy.OutputField(desc="The sentiment: positive, negative, or neutral")


# Define a module that uses the signature
# Modules are composable building blocks that use signatures
class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought adds reasoning before producing output
        self.analyze = dspy.ChainOfThought(Sentiment)

    def forward(self, text):
        # Execute the signature with the input
        result = self.analyze(text=text)
        return result.sentiment


# Example usage
if __name__ == "__main__":
    # Configure DSPy with a language model
    lm = dspy.LM("openrouter/openai/gpt-3.5-turbo")
    dspy.configure(lm=lm)

    # Create and use the module
    analyzer = SentimentAnalyzer()

    # Test it
    text = "I absolutely love this product! It's amazing!"
    sentiment = analyzer(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
