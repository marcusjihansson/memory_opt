import dspy
from dspy.predict.rlm import RLM


class QuestionAnswer(dspy.Signature):
    """Answer questions based on a long document context."""

    context = dspy.InputField(desc="The document or long text to analyze")
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="A comprehensive answer based on the context")


class ContextualReasoning(dspy.Signature):
    """Advanced reasoning over long documents with chain of thought."""

    context = dspy.InputField(desc="The document content")
    question = dspy.InputField(desc="Question requiring deep analysis")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
    answer = dspy.OutputField(desc="Final answer derived from reasoning")


class SimpleRLMQA(dspy.Module):
    """Basic Q&A module using RLM for long context handling."""

    def __init__(self):
        super().__init__()

        self.rlm = RLM(signature="context, query -> answer", max_iterations=10)

    def forward(self, context, question):
        """Process question over long context using RLM."""
        result = self.rlm(context=context, query=question)

        answer = result.answer if hasattr(result, "answer") else str(result)
        return dspy.Prediction(answer=answer if answer else "Unable to generate answer")


class ProductionRLMQA(dspy.Module):
    """Production-ready Q&A system with error handling and logging."""

    def __init__(self):
        super().__init__()

        self.rlm = RLM(signature="context, question -> answer", max_iterations=15)

        self.fallback = dspy.ChainOfThought(QuestionAnswer)

    def forward(self, context, question, use_fallback=True):
        """Production forward pass with error handling."""
        try:
            result = self.rlm(context=context, question=question)
            answer = result.answer if hasattr(result, "answer") else str(result)

            if answer:
                return dspy.Prediction(answer=answer, method="rlm", success=True)
            elif use_fallback:
                fallback_result = self.fallback(
                    context=context[:8000], question=question
                )
                return dspy.Prediction(
                    answer=fallback_result.answer, method="fallback", success=True
                )
            else:
                return dspy.Prediction(
                    answer="Unable to process query", method="none", success=False
                )

        except Exception as e:
            if use_fallback:
                fallback_result = self.fallback(
                    context=context[:8000], question=question
                )
                return dspy.Prediction(
                    answer=fallback_result.answer,
                    method="fallback_error",
                    success=True,
                    error=str(e),
                )
            else:
                return dspy.Prediction(
                    answer=f"Error: {str(e)}",
                    method="error",
                    success=False,
                    error=str(e),
                )
