import json
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
import re


def replace_patterns(input_string):
    # Define the patterns and replacements
    patterns_replacements = {
        r"': \{'": r'": {"',
        r"'}, '": r'"}, "',
        r"': '": r'": "',
        r"', '": r'", "',
        r"'}}": r'"}}',
        r"\{'": r'{"',
        r"\\'s ": r"'s",
        r"'}": r'"}',
        r"':": r'":',
        r", '": r', "'
    }

    # Apply each pattern replacement
    for pattern, replacement in patterns_replacements.items():
        input_string = re.sub(pattern, replacement, input_string)

    return input_string


def run_query(rag_chain, mpq_dict, retriever):
    query = "Provide a  RFC8259 compliant JSON response following this format without \
    deviation. If there is no relevant information in the transcript, keep the original \
    JSON structure, but fill in the field with 'N/A'. Important: DO NOT provide any \
    information in a field that is not directly relevant. Here is the form: "

    # Function to retrieve context
    def retrieve_context(query):
        return retriever.get_relevant_documents(query)

    trio_list = []
    print("running query...")
    final_json = {}
    for item in mpq_dict.keys():
        if(item != "CognitivelyImpairedandIntellectualDisability.json"):
            continue
        print(item)
        # if(item == "SummaryofEventsLeadingUpToandFollowingMP'sDisappearance.json"):
        response = rag_chain.invoke(query + str(mpq_dict[item]))
        context_documents = retriever.get_relevant_documents(query + str(mpq_dict[item]))
        # print(query + str(mpq_dict[item]))
        # print(len(context_documents))
        # print(str(context_documents))
        context = [doc.page_content for doc in context_documents]
        # form_json
        print("pre: " + response)
        print("post: " + replace_patterns(response))
        response_json = json.loads(replace_patterns(response))
        print(response_json)
        final_json[item.removesuffix('.json')] = response_json

        for field_name, field_value in response_json.items():
            if (field_value == "N/A" or isinstance(field_value, dict)):
                continue
            trio_list.append([mpq_dict[item][field_name], field_value, context])

    print(response_json)
    print(trio_list)
    final_json = json.loads(json.dumps(final_json, indent=4))

    # print(len(trio_list))
#
# for query in trio_list:
#   print(query[0])
    return final_json, trio_list
