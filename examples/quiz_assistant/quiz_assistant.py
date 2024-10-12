import logging
import os
from pydantic import BaseModel
import outspeed as sp
import aiohttp



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
    questions_to_ask = []
    logging.info("INITIALIZING SEARCHTOOL")
    # EMBEDDING STUFF here


    async def run(self, user_response: UserResponse) -> SearchResult:
        SearchTool.chat_history.append(user_response)
        logging.info(f"search_tool: {user_response}")
        return SearchResult(result = "Ask the user about dinosaurs right now")

@sp.App()
class VoiceBot:
    async def setup(self) -> None:
        # Initialize the AI services
        self.deepgram_node = sp.DeepgramSTT(sample_rate=8000)
        self.llm_node = sp.OpenAIRealtime(
            system_prompt="You are an assistant that asks a user questions.",
            tools=[SearchTool(name="answer_tool", description="this tool is called whenenver the user says something to provide context for the response.", parameters_type=UserResponse, response_type=SearchResult)],
            #tool_choice="required" # TODO: figure out why turning this on causes everything to break
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
