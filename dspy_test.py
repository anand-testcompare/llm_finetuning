from dspy import Module, Program, evaluate

# Sample Question Answering Dataset (replace with your actual data)
train_data = [
  ("What is the capital of France?", "Paris"),
  ("What is the largest planet in our solar system?", "Jupiter"),
  ("What is the meaning of life?", "42"),  # This might be a tricky one for the LLM ;)
]

# Split data into training and validation sets (80%/20% split here)
train_len = int(0.8 * len(train_data))
train_set, validation_set = train_data[:train_len], train_data[train_len:]


# Mock LLM function (replace with your actual LLM interaction logic)
def get_llm_answer(question):
  # Simulate LLM response based on a simple matching logic (not ideal, for demonstration purposes only)
  for q, a in train_data:
    if question.lower() == q.lower():
      return a
  return "Sorry, I couldn't find an answer in my knowledge base."

# Define Evaluation Metric (Accuracy in this case)
def accuracy(predictions, ground_truth):
  correct = 0
  for p, gt in zip(predictions, ground_truth):
    if p == gt:
      correct += 1
  return correct / len(predictions)


class QAModule(Module):
  def __init__(self, lm):
    super().__init__(name="qa")
    self.lm = lm

  def signature(self):
    return ("question", "answer")

  def forward(self, question):
    answer = self.lm(question)
    return answer


class QAProgram(Program):
  def __init__(self, lm):
    super().__init__()
    self.qa_module = QAModule(lm)
    self.add_module(self.qa_module)

  def run(self, question):
    answer = self.qa_module(question)
    return answer

  def evaluate(self, validation_set):
    # Get predictions for validation set
    predictions = [self.run(q) for q, _ in validation_set]
    # Get ground truth labels from validation set
    ground_truth = [a for _, a in validation_set]
    # Calculate accuracy
    acc = accuracy(predictions, ground_truth)
    return acc


# Create program with mock LLM
program = QAProgram(get_llm_answer)

# Train model (simulated here by training data exposure)
# In reality, training would involve feeding the LLM large amounts of data

# Evaluate model on validation set
accuracy = program.evaluate(validation_set)
print(f"Model Accuracy on Validation Set: {accuracy:.2f}")

# Ask a question (use test data or unseen data for better evaluation)
question = "What is the airspeed velocity of an unladen swallow?"
answer = program.run(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
