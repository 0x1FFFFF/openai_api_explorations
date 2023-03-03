import glob
import openai
import os

from pydub import AudioSegment 
from pydub.utils import make_chunks

openai.api_key = os.getenv("OPENAI_API_KEY")


def divide_audio_files_chunks(file_path):
    myaudio = AudioSegment.from_file(file_path, "mp3") 
    chunk_length_ms = 10 * 60 * 1000 # 10 min in ms
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of 100 secounds
    chunk_dir = './assets/chunked/' + file_path + "/"
    if not os.path.exists(chunk_dir):
        os.makedirs(chunk_dir) 
        for i, chunk in enumerate(chunks):
            print(f"spliting file {file_path} into chunks of {chunk_length_ms} ms")
            chunk_name = chunk_dir + "segment_{0}.mp3".format(i) 
            print ("exporting", chunk_name) 
            chunk.export(chunk_name, format="mp3")
    else:
        print("Chunk directory already exists")
    return chunk_dir
        

def transcribe_audio_file(file_path):
    chunk_dir = divide_audio_files_chunks(file_path=file_path)
    chunk_list = sorted([file for file in glob.glob(chunk_dir + "*.mp3")])
    transcript_output = ""
    transcript_summary_output = ""
    if chunk_list:
        print("Transcribing audio files")
        for index, chunk in enumerate(chunk_list):
            print(f"Transcribing file: {chunk}")
            audio_file= open(chunk, "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcript_len = len(transcript.text)
            print(f"Finsihed transcribing file: {chunk}, f{index} part with 10 minutes of the original audio, output length is for this file is {transcript_len}")
            transcript_output += transcript.text + "\n" + "-----------------------10 minutes of audio breaker-----------------------" + "\n"

    output_dir = './output/'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    transcript_file_path = "output/" + file_path.split("/")[1].split(".")[0] + "_transcript.txt"
    with open(transcript_file_path, "w+") as f:
        f.write(transcript_output)
    return transcript_file_path

def summarize_transcript(transcript_output_file):
    list_of_words = []
    final_summary_list = [f"Summary of {transcript_output_file}"]
    with open(transcript_output_file, "r") as f:
        list_of_words = f.read().split()
    word_limit = 2000

    for i in range(0, len(list_of_words)-word_limit, word_limit):
        print(f"Generating summary for {word_limit} words")
        transcript_part = " ".join(list_of_words[i:i+word_limit])
        prompt = f"Take this part of meeting trascript: \n\n{transcript_part}, \n\n and summarize the important information"
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        # print(response)
        final_summary_list += response.choices[0].text
    
    summary_output_file = transcript_output_file.split("_")[0] + "_summary.md"
    with open(summary_output_file, "w+") as f:
        f.write("".join(final_summary_list))
    
    with open(summary_output_file, "r+") as f:
        split_summary = f.read()
        prompt = f"Summarize this meeting transcript: \n\n{split_summary}, \n\n and summarize it with important details"
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        f.write(f"\n\n##Final Summary\n{response.choices[0].text}",)

    return summary_output_file

if __name__ == '__main__':
    input_file = "assets/test.mp3"
    trascript_path = transcribe_audio_file(input_file)
    summarize_transcript(trascript_path)