import json
import os
import sys
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_NAME, DATA_DIR

def finetune_model(qa_file='qa_pairs.json', output_path='models/fine_tuned_model', epochs=1):
    """
    Fine-tunes the SentenceTransformer model on a QA dataset.
    """
    qa_path = os.path.join(DATA_DIR, qa_file)
    if not os.path.exists(qa_path):
        print(f"Dataset not found at {qa_path}")
        return

    # Load Data
    with open(qa_path, 'r') as f:
        data = json.load(f)

    train_examples = []
    for item in data:
        train_examples.append(InputExample(texts=[item['question'], item['answer']]))

    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Load Model
    model = SentenceTransformer(MODEL_NAME)

    # Define Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    print(f"Starting fine-tuning for {epochs} epochs...")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, show_progress_bar=True)

    # Save
    full_output_path = os.path.join(os.path.dirname(DATA_DIR), output_path)
    model.save(full_output_path)
    print(f"Model saved to {full_output_path}")

if __name__ == "__main__":
    finetune_model()
