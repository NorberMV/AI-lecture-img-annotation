# # This will call the desired graph
from graph.test_graph import run_testing_graph

IMAGE_PATH = "tool_utils/tomatoes_dataset/garden-photos-4.jpg"
# TEST USER QUERIES FOR THE GRADIO DEMO:
# user_query = "Can you show me the ripened tomatoes?" # WORKING
user_query = "Can you show me the fully ripened tomatoes?" # WORKING !
# user_query = "Can you select all the tomatoes?" # WORKING !
# user_query = "Hi there! who are you?" # WORKING

if __name__ == "__main__":
    run_testing_graph(query=user_query)
    # run_graph()


