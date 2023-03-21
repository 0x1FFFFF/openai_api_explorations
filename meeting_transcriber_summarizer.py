import glob
import openai
import os
import re
import whisper
import pprint

from pydub import AudioSegment 
from pydub.utils import make_chunks
from pyannote.audio import Pipeline

openai.api_key = os.getenv("OPENAI_API_KEY")
total_cost = 0
total_prompt_tokens = 0
total_completion_tokens = 0
token_to_price = {
    'gpt-3.5-turbo': 0.000002,
    'gpt-4-8k': 0,
    'gpt-4-32k': 0
}
backend_model = 'gpt-3.5-turbo'

whisper_price_per_minute = 0.006
# ---------------------Experimental stuff-------------------------

# hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
# pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=hugging_face_token)
# spacermilli = 2000
# spacer = AudioSegment.silent(duration=spacermilli)

# print("Initialized OpenAI and Pyannote")

# def diarize_audio_file(input_file_path):
#     dz_config = {
#         'uri': 'test',
#         'audio': input_file_path
#     }
#     dz_segments = pipeline(dz_config)

#     with open('output/' + input_file_path.split("/")[1].split(".")[0] + "_dz.txt", "w+") as f:
#         f.write(str(dz_segments))

# def generate_transcript_with_diarization(dz_txt_path):
#     dz = open(dz_txt_path).read().splitlines()
#     dzList = []
#     for l in dz:
#         start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
#         start = millisec(start) - spacermilli
#         end = millisec(end)  - spacermilli
#         lex = not re.findall('SPEAKER_01', string=l)
#         dzList.append([start, end, lex])

#     return dzList

# def modify_audio_with_spacer(input_audio_path):
#     sounds = spacer
#     segments = []
#     audio = AudioSegment.from_mp3(input_audio_path)
#     dz_txt_path = 'output/' + input_audio_path.split("/")[1].split(".")[0] + "_dz.txt"
#     dz = open(dz_txt_path).read().splitlines()
#     for l in dz:
#         start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
#         start = int(millisec(start)) #milliseconds
#         end = int(millisec(end))  #milliseconds
        
#         segments.append(len(sounds))
#         sounds = sounds.append(audio[start:end], crossfade=0)
#         sounds = sounds.append(spacer, crossfade=0)

#     dz_output = "output/" + input_audio_path.split("/")[1].split(".")[0] + "_diarization.wav"
#     sounds.export(f"{dz_output}", format="wav") #Exports to a wav file in the current path.

# def millisec(time_str):
#     spl = time_str.split(":")
#     s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
#     return s

# def whisper_transcribe_audio(audio_file):
#     model = whisper.load_model("large")
#     result = model.transcribe(audio_file, language="zh")
#     with open("output/whisper_large_transcript.txt", "w+") as f:
#         f.write(result.text)
# --------------------------------------------------------------------------------------------------

def divide_audio_files_chunks(file_path):
    global total_cost
    myaudio = AudioSegment.from_file(file_path, os.path.splitext(file_path)[1][1:]) 
    anticipated_whisper_cost = round(myaudio.duration_seconds/60) * whisper_price_per_minute
    print(f"Anticipated whisper cost for file {file_path} is {anticipated_whisper_cost} USD")
    total_cost += anticipated_whisper_cost
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
        

def transcribe_audio_file(file_path, language="English"):
    chunk_dir = divide_audio_files_chunks(file_path=file_path)
    chunk_list = sorted([file for file in glob.glob(chunk_dir + "*.mp3")])
    transcript_output = ""
    if chunk_list:
        print("Transcribing audio files")
        for index, chunk in enumerate(chunk_list):
            print(f"Transcribing file: {chunk}")
            audio_file= open(chunk, "rb")
            transcript = openai.Audio.transcribe(
                "whisper-1", 
                audio_file,
                prompt="Transcribe the audio with each line being a sentence by a speaker, also annotate this sentence with corresponding timestamp of the audio recording")
            transcript_len = len(transcript.text)
            print(transcript.text)
            print(f"Finsihed transcribing file: {chunk}, part f{index} with 10 minutes of the original audio, output length is for this file is {transcript_len}")
            transcript_output += transcript.text + "\n" + "-----------------------10 minutes of audio breaker-----------------------" + "\n"

    output_dir = './output/'
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    transcript_file_path = "output/" + file_path.split("/")[1].split(".")[0] + "_transcript.txt"
    with open(transcript_file_path, "w+") as f:
        f.write(transcript_output)
    return transcript_file_path

# to work around the limitation of 4096 max token
def summarize_chunk_iterator(input_file_path, output_file_path, translate_zh=False, language="English", word_limit=300, override_part_limit=0):
    list_of_words = []
    part_word_limit = 0
    with open(output_file_path, "a") as sum_file:
        with open(input_file_path, "r") as f:
            if language == "English":
                list_of_words = f.read().split()
                part_word_limit = 3000
            elif language == "Chinese":
                list_of_words = [c for c in f.read()]
                part_word_limit = max(override_part_limit, 1500)
            # print(len(list_of_words))
            part_number = 1
            sum_file.write(f"# Summary of {input_file_path}\n\n")
            for i in range(0, len(list_of_words)-part_word_limit, part_word_limit):            
                # print(f"Generating part of summary")
                sum_file.write(f"\n\n## Part{part_number}\n\n")
                transcript_part = " ".join(list_of_words[i:i+part_word_limit])
                summary_part = summarize_text(transcript_part, language=language, word_limit=word_limit)
                # print(f"Writing part of summary", len(summary_part))
                # pprint.pprint(summary_part)
                sum_file.write(summary_part)
                # print("Finished writing part of summary")
                part_number += 1

    print("iterator finished")
    return output_file_path

def summarize_transcript(transcript_output_file, translate_zh=False, language="English", word_limit=300):
    print("Summarizing transcript")
    summaries = []
    summary_output_file = transcript_output_file.split("_")[0] + "_summary.md"
    summarize_chunk_iterator(transcript_output_file, summary_output_file, translate_zh=translate_zh, language=language, word_limit=word_limit)
    condensed_summary(summary_output_file, translate_zh=translate_zh, language=language)
    return summary_output_file

def condensed_summary(summary_output_file, translate_zh=False, language="English", word_limit=300):
    print("Condensing summary")
    condensed_summary_output_file = summary_output_file.split("_")[0] + "_condensed_summary.md"
    summarize_chunk_iterator(summary_output_file, condensed_summary_output_file, translate_zh=translate_zh, language=language, word_limit=word_limit, override_part_limit=2000)
    return condensed_summary_output_file

def summarize_text(text, language="English", word_limit=300):
    global total_cost
    global total_completion_tokens
    global total_prompt_tokens

    response = openai.ChatCompletion.create(
        model=backend_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps me summarize my meeting transcripts."},
            {"role": "user", "content": f"Here is a part of meeting notes I have: {text}"},
            {"role": "user", "content": f"Now summarize this part for me with great details and in a professional tone with strictly less than {word_limit} words in {language}. Please only capture the important details of the main topics of the meetings in bullet points and leave out the small talk and irrelevant chats."},
        ]
    )
    usage = response.usage
    print(usage)
    cost = usage['prompt_tokens'] * token_to_price[backend_model] + usage['completion_tokens'] * token_to_price[backend_model]
    total_cost += cost
    total_completion_tokens += usage['completion_tokens']
    total_prompt_tokens += usage['prompt_tokens']
    print(f"This message uses {usage['prompt_tokens']} prompt_tokens and {usage['completion_tokens']} completion_tokens, this message cost ${cost}, running total cost: ${total_cost}")
    return response.choices[0].message.content

def translate_text(text, language="English"):
    response = openai.ChatCompletion.create(
        model=backend_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that helps me translate text to {language}"},
            {"role": "user", "content": f"Here is some text I need you to translate in accurate details: {text}"},
        ]
    )
    return response.choices[0].message.content
        
if __name__ == '__main__':
    # input_file = "assets/test2_zh.mp3"
    # transcribe_audio_file(input_file)
    # summarize_transcript(transcribe_audio_file(input_file))
    # summarize_transcript("output/2023.3.21 Management Meeting with 泽森科工.txt", language="Chinese")
    condensed_summary("output/2023.3.21 Management Meeting with 泽森科工.txt_summary.md", language="Chinese")
    print(f"Done! Total Prompt Tokens: {total_prompt_tokens}. Total Completion Tokens: {total_completion_tokens}. Total cost: ${total_cost}")
    # diarize_audio_file(input_file)
    # modify_audio_with_spacer(input_file)
    # whisper_transcribe_audio("output/test2_zh_diarization.wav")