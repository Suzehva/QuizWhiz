import logging
import os
from pydantic import BaseModel
import outspeed as sp
import aiohttp
import voyageai
import numpy as np
#import fitz
import time
import pickle
import re

from embedding_helper import extract_text, chunk_text, embed_texts, extract_text_from_pdf, generate_questions

def check_outspeed_version():
    import importlib.metadata

    from packaging import version

    required_version = "0.1.147"

    try:
        current_version = importlib.metadata.version("outspeed")
        if version.parse(current_version) < version.parse(required_version):
            raise ValueError(f"Outspeed version {current_version} is not greater than {required_version}.")
        else:
            print(f"Outspeed version {current_version} meets the requirement.")
    except importlib.metadata.PackageNotFoundError:
        raise ValueError("Outspeed package is not installed.")


check_outspeed_version()
# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)

"""
The @outspeed.App() decorator is used to wrap the VoiceBot class.
This tells the outspeed server which functions to run.
"""


class UserResponse(BaseModel):
    user_response: str


class SearchResult(BaseModel):
    result: str



class SearchTool(sp.Tool):
    chat_history = []
    questions_and_context = []
    
    
    def __init__(self, name, description, parameters_type, response_type):
        super().__init__(name=name, description=description, parameters_type=parameters_type, response_type=response_type)
        logging.info("INITIALIZING SEARCHTOOL")
    
        # EMBEDDING STUFF here
        text_path = input("Please upload your PDF file (enter the path): ")
        # /Users/suze/Documents/Storing_code/outspeed/examples/quiz_assistant/basic-plant-care-understanding-your-plants-needs-hla-6461.pdf
        while not os.path.isfile(text_path) or not text_path.endswith('.pdf'):
            print("Invalid file. Please ensure the path points to a PDF file.")
            text_path = input("Please upload your PDF file (enter the path): ")
        text = extract_text_from_pdf(text_path)
        #questions_to_ask = generate_questions(text)
        #logging.info(f"QUESTIONS_TO_ASK: {questions_to_ask}")
        documents = chunk_text(extract_text(text_path))
        embeddings = embed_texts(documents)

        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f) # maybe unnecessary
        vo = voyageai.Client(api_key="pa-gA94dVkc9_oN6GXJaheWdjCdrzwY06JoNOCDbqyBkqg")
        questions_to_ask = ['1. What do plants need to survive?',  'How can mulch be beneficial for a garden?', "What is biennial?"]
        query_embeddings = vo.embed(questions_to_ask, model="voyage-3", input_type="query").embeddings

        
        for i, query_embedding in enumerate(query_embeddings):
            similarity = np.dot(embeddings, query_embedding)
            retrieved_id = np.argmax(similarity)
            info_dict = {
                'question': questions_to_ask[i],
                'score':0,
                'context':documents[retrieved_id],
            }
            SearchTool.questions_and_context.append(info_dict)
            time.sleep(5)


    async def run(self, user_response: UserResponse) -> SearchResult:
        SearchTool.chat_history.append(user_response)
        logging.info(f"search_tool: {user_response}")
        

        # TODO: generate new questions based on chat history

        # take a question from question bank
        if len(SearchTool.questions_and_context) > 0:
            info_dict = SearchTool.questions_and_context.pop()
            output = f"Ask the user the following question: {info_dict['question']}. Provide the user feedback on their context: {info_dict['context']} DO NOT GIVE THE USER THIS CONTEXT. ONLY USE IT TO PROVIDE FEEDBACK AFTER THEY RESPOND!"
        else:
            output = 'Tell the user you ran out of questions to ask'
            

        return SearchResult(result = output)

@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.deepgram_node = sp.DeepgramSTT(sample_rate=8000)
        self.llm_node = sp.OpenAIRealtime(
            system_prompt="You are an assistant that asks a user questions.",
            tools=[SearchTool(
                name="answer_tool", 
                description="call this tool every time you want to answer a user or you start a chat", 
                parameters_type=UserResponse, 
                response_type=SearchResult, 
            )],
            tool_choice="required"
        )
         

    @sp.streaming_endpoint()
    async def run(self, audio_input_queue: sp.AudioStream, text_input_queue: sp.TextStream) -> sp.AudioStream:
        # Set up the AI service pipeline
        # audio_output_stream: sp.AudioStream
        # transcript_queue = self.deepgram_node.run(audio_input_queue)
        # question_instruction_queue = sp.map(transcript_queue, self.process)

        # # testing purposes
        # text_queue = self.deepgram_node.run(text_input_queue)
        # question_instruction_queue = sp.map(text_queue, self.test)

        audio_output_stream, text_output_queue = self.llm_node.run(text_input_queue, audio_input_queue)

        return audio_output_stream, text_output_queue

    async def teardown(self) -> None:
        """
        Clean up resources when the VoiceBot is shutting down.

        This method is called when the app stops or is shut down unexpectedly.
        It should be used to release resources and perform any necessary cleanup.
        """
        await self.llm_node.close()


if __name__ == "__main__":
    # Start the VoiceBot when the script is run directly
    VoiceBot().start()
