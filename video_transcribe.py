import os
import sys
import subprocess
import time
from pathlib import Path
import whisper
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov"]

spinner_cycle = ["|", "/", "-", "\\"]

def spinner(prefix, run_func, *args, **kwargs):
    """
    Show spinner while running a function or subprocess.
    Updates the same line in place.
    """
    done = False
    result = None

    def target():
        nonlocal result
        result = run_func(*args, **kwargs)

    import threading
    t = threading.Thread(target=target)
    t.start()

    i = 0
    while t.is_alive():
        sys.stdout.write(f"\r{prefix} {spinner_cycle[i % len(spinner_cycle)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    t.join()
    sys.stdout.write(f"\r{prefix}\n")
    sys.stdout.flush()
    return result

def transcribe_audio(model, audio_file, transcript_file):
    result = model.transcribe(str(audio_file), language="en", verbose=True)
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return transcript_file

def process_video(video_path, whisper_model, generate_pdf=False, llm_model="gemma3"):
    base = Path(video_path).stem
    dir_path = Path(video_path).parent

    audio_file = dir_path / f"{base}.mp3"
    transcript_file = dir_path / f"{base}.txt"
    study_file = dir_path / f"{base}_study.md"
    pdf_file = dir_path / f"{base}.pdf"

    print(f"\n{video_path.name}")

    steps = ["video"]

    # Step 1: Extract audio
    if not audio_file.exists():
        spinner("    video > ...", subprocess.run,
                ["ffmpeg", "-i", str(video_path),
                 "-vn", "-c:a", "libmp3lame", "-q:a", "2",
                 str(audio_file)], check=True)
    steps.append("audio")

    # Step 2: Transcribe
    if not transcript_file.exists():
        spinner("    video > audio > ...", transcribe_audio,
                whisper_model, audio_file, transcript_file)
    steps.append("transcript")

    # Step 3: Study material (directly from transcript)
    if not study_file.exists():
        def study():
            study_prompt = PromptTemplate.from_template(
                "Using the following transcript, generate structured study material including:\n"
                "- Key concepts and definitions\n"
                "- Bullet-point notes\n"
                "- Glossary of important terms with explanations (format as a bullet list, NOT a table)\n"
                "- 25 practice questions (mix of MCQ and short answer)\n\n{transcript}"
            )
            llm = OllamaLLM(model=llm_model)
            study_chain = RunnableSequence(first=study_prompt, last=llm)
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            study_material = study_chain.invoke({"transcript": transcript_text})
            with open(study_file, "w", encoding="utf-8") as f:
                f.write(study_material)
        spinner("    video > audio > text > ...", study)
    steps.append("study guide")

    # Step 4: PDF
    if generate_pdf:
        if not pdf_file.exists():
            # Path to the LaTeX header file
            header_file = Path(__file__).parent / "header.tex"

            pandoc_command = [
                "pandoc", str(study_file),
                "-o", str(pdf_file),
                "--pdf-engine=pdflatex",
                f"--include-in-header={str(header_file)}",
                "--variable", "fontsize=12pt",
                "--toc",  # Table of contents
                "--toc-depth=3",
                "--number-sections"
            ]

            spinner("    video > audio > text > markdown > PDF ...", subprocess.run,
                    pandoc_command, check=True)
        steps.append("PDF")

    # Final pipeline line
    print("    " + " > ".join(steps))


def process_directory(directory, whisper_model, generate_pdf=False):
    dir_path = Path(directory)
    for file in dir_path.iterdir():
        if file.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                process_video(file, whisper_model, generate_pdf=generate_pdf)
            except Exception as e:
                print(f"    [ERROR] Could not process {file.name}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process videos into study material/PDFs")
    parser.add_argument("directory", help="Path to folder containing video files")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF output (default: skip)")
    args = parser.parse_args()

    print("AI is warming up... ready to crunch some knowledge.")

    whisper_model = whisper.load_model("medium")

    process_directory(args.directory, whisper_model, generate_pdf=args.pdf)
