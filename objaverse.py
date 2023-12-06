import objaverse
import multiprocessing

uids = [
    "af2ac66229dd4429953a9e5b9fc2fdc1",
    # "ce1ad6c08a8c4e2f8c08eb910a0f8e1b",
    # "f3dd3064db4f4e8880344425970cecad",
    # "92db5e13fa0c4c27a2689b962fc6305f",
]

if __name__ == "__main__":
    annotations = objaverse.load_annotations(uids)
    # Check if all the models are downloadable
    for uid in uids:
        if annotations[uid]["isDownloadable"] == False:
            print("Model {} is not downloadable".format(uid))
    # Follow the instructions in https://colab.research.google.com/drive/1ZLA4QufsiI_RuNlamKqV7D7mn40FbWoY?usp=sharing#scrollTo=sgCHLqFVICGL
    processes = multiprocessing.cpu_count()
    download_path = objaverse.load_objects(
        uids, download_processes=processes
    )
    print(download_path)