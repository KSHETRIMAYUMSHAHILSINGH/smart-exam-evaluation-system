from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training_dataset.csv"

# Extra synthetic examples focused on the weak 4–7 mark range.
# These intentionally use paraphrases/synonyms instead of copying rubric words.
EXTRA = {
    "Q1": [
        (4, "AI lets software do some tasks that usually need people, such as choosing between options and identifying objects in pictures."),
        (4, "Artificial intelligence makes computers behave intelligently by using data to improve what they do. It is used in virtual assistants."),
        (5, "AI enables machines to learn from data and make decisions. Examples include chatbots and image recognition."),
        (5, "Artificial intelligence allows computer programs to adapt from examples and handle tasks such as recommendations and speech-based assistants."),
        (6, "AI is a field of computing that builds systems able to learn from information, decide between alternatives, and identify useful structures in data. It is used in chatbots and recommendation tools."),
        (6, "Artificial intelligence gives machines abilities similar to people, including learning from experience, interpreting language, and making automated decisions."),
        (7, "AI enables software to perform human-like cognitive tasks. Systems can learn from data, infer suitable actions, solve tasks, interpret language, and identify objects or trends. Common uses include chatbots and recommendation systems."),
        (7, "Artificial intelligence develops computer systems that can adapt from examples, draw conclusions, handle new problems, understand text or speech, and detect meaningful structures in data."),
    ],
    "Q2": [
        (4, "Machine learning lets a program improve by looking at examples instead of relying only on fixed instructions."),
        (4, "ML uses past data to find useful relationships that can help a computer make future choices."),
        (5, "Machine learning trains a model on examples so it can make predictions on new data without writing a separate rule for every case."),
        (5, "ML systems study training examples, adjust their internal parameters, and then use what they learned on unseen inputs."),
        (6, "Machine learning is part of AI where algorithms discover relationships in data and use them for predictions or decisions. The model improves during training rather than being fully hard-coded."),
        (6, "In ML, examples are used to fit a model. The fitted model can then classify items or estimate values for new cases."),
        (7, "Machine learning allows software to learn statistical relationships from training data, adjust a model from examples, and generalize those relationships to unseen inputs for classification, regression, or prediction."),
        (7, "ML builds predictive models from data rather than manually programming every decision rule. Training updates the model, which is later used to classify, estimate, or choose actions for new inputs."),
    ],
    "Q3": [
        (4, "NLP helps computers work with things people write or say, such as messages, documents, and speech."),
        (4, "Natural language processing is used when software needs to read text, understand a user's words, or generate a response."),
        (5, "NLP enables machines to process text and speech. It is used in translation, chatbots, and detecting opinions in text."),
        (5, "Natural language processing applies AI techniques to human communication so systems can classify text, answer questions, or translate between languages."),
        (6, "NLP is an AI field for analyzing and interpreting written or spoken language. It supports tasks such as sentiment detection, translation, question answering, and comparing the meaning of two texts."),
        (6, "Natural language processing converts human language into representations that software can analyze for tasks such as classification, translation, and text matching."),
        (7, "NLP enables computers to analyze human language, represent meaning, classify or translate text, answer questions, and compare how similar two responses are. These techniques can also support automatic answer evaluation."),
        (7, "Natural language processing combines language analysis with machine learning so computers can interpret text or speech, detect sentiment, translate content, answer questions, and assess semantic similarity."),
    ],
    "Q4": [
        (4, "Cloud computing means using computing resources through the internet instead of keeping all hardware and services on one local machine."),
        (4, "The cloud lets users access remote computing services online, for example file storage and hosted applications."),
        (5, "Cloud computing provides remote servers, storage, and software over the internet, so organizations do not need to own every system locally."),
        (5, "With cloud services, users can access computing resources online and increase or reduce those resources when needed."),
        (6, "Cloud computing delivers servers, databases, storage, networking, and software remotely. Resources can be adjusted according to demand and accessed from different locations."),
        (6, "The cloud provides shared online infrastructure that can scale with usage, improve access from different devices, and reduce the need to maintain all physical equipment in-house."),
        (7, "Cloud computing supplies computing infrastructure and software through the internet. Organizations can use remote servers, databases, storage, and networks, scale capacity with demand, and reduce local hardware maintenance."),
        (7, "Cloud services make computing resources available remotely on demand. They support flexible scaling, access from many locations, and lower infrastructure management requirements for users and organizations."),
    ],
    "Q5": [
        (4, "A DBMS is software that keeps information organized so applications can save it, find it, and change it."),
        (4, "Database management software provides a structured place for records and lets users search or update those records."),
        (5, "A DBMS stores and organizes data and provides commands for retrieving and updating it. It can also control who is allowed to access information."),
        (5, "Database management systems help applications maintain structured records, run queries, and avoid unmanaged duplicate copies of information."),
        (6, "A DBMS manages structured data, supports efficient queries and updates, and helps keep stored information accurate and consistent. It can also restrict access to authorized users."),
        (6, "Database management software provides organized storage, retrieval, updates, access control, and mechanisms that help preserve reliable data for multiple users."),
        (7, "A DBMS is software for managing structured collections of data. It supports efficient storage and querying, protects access, maintains reliable and consistent records, and allows multiple applications or users to work with the database."),
        (7, "Database management systems organize and retrieve records, enforce access rules and data quality, reduce unnecessary duplication, and support many users. They are widely used in banking, hospitals, education, and online stores."),
    ],
}

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Run scripts/build_training_dataset.py first.")

    df = pd.read_csv(DATA)

    # Make the script safe to run more than once.
    if "augmentation_batch" in df.columns:
        df = df[df["augmentation_batch"].fillna("") != "midrange_v2"].copy()
    else:
        df["augmentation_batch"] = ""

    rows = []
    for qid, examples in EXTRA.items():
        base = df[df["question_id"] == qid]
        if base.empty:
            raise ValueError(f"Question {qid} was not found in the dataset.")
        first = base.iloc[0]

        for score, answer in examples:
            rows.append({
                "question_id": qid,
                "question_text": first["question_text"],
                "model_answer": first["model_answer"],
                "student_answer": answer,
                "human_score": float(score),
                "rubric_keywords": first["rubric_keywords"],
                "source": "synthetic_augmented",
                "augmentation_batch": "midrange_v2",
            })

    extra_df = pd.DataFrame(rows)

    # Ensure the existing rows have the new column.
    if "augmentation_batch" not in df.columns:
        df["augmentation_batch"] = ""

    out = pd.concat([df, extra_df], ignore_index=True)
    out.to_csv(DATA, index=False)

    print(f"Added {len(extra_df)} mid-range paraphrased examples.")
    print(f"Dataset now has {len(out)} rows.")
    print("\nScore distribution:")
    print(out["human_score"].value_counts().sort_index())
    print("\nPer question:")
    print(out.groupby("question_id").size())


if __name__ == "__main__":
    main()
