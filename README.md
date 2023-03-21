## Experimenting with Open AI's Public Dev API
- `meeting_transcriber_summarizer.py`: Implements a simple meeting transciber and summarizer. Takes in an audio recording and splits into chunks and sends to `Whisper AI` then summarized with the all new and much cheaper `gpt-3.5-turbo`.

## To Install:
- Install poetry
- Run `poetry inistall`
- `poetry shell`
- Play on your own


## Meeting Transcriber and Summarizer To-do:
- [ ] Multiprocessing - Right now all steps are sequential. This is a bottleneck.
  - [ ] Audio Transcribe
  - [ ] Meeting Part Summary
- [ ] Decouple and make modules
- [ ] Configurable detail level of part summary
- [ ] Configurable detail level of condensed summary
- [ ] Polish output file formatting
  - [ ] Enable output format
    - [ ] PDF
    - [ ] HTML
    - [ ] Markdown
- [ ] User-friendly IO for non-technical users -- Accept audio file and output summary files
  - [ ] Web App
  - [ ] Packaged Desktop Apps
    - [ ] Mac
    - [ ] Windows