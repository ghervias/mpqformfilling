import pandas as pd
from datasets import Dataset

def ragas_eval(trio_list):
    questions = []
    answers = []
    context_list = []
    ground_truths = ["The person being interviewed realized that the MP was missing after getting home from Costco after 2 hours.",
                     "For two hours, the person being interviewed drove around the neighborhood and an area further beyond the neighborhood looking for the MP. They looked at a nearby park and a strip mall.",
                     "The MP did not leave any notes as far as the person being interviewed knows, but he did not search the bedroom.", "The MP most likely has his watch.",
                     "The MP has been mourning his wife's death. Recently, the MP has been more active.", "The MP's wife died recently. The MP may also have fallen back into a gambling addiction."]


    for query in trio_list:
      questions.append(query[0])
      answers.append(query[1])
      context_list.append(query[2])


    print(context_list)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": context_list,
        "ground_truth": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )



    result = evaluate(
        dataset = dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()

    pd.display(df)