import gradio as gr

from gradio_client import Client, file

import os
import requests
import shutil
import xml.etree.ElementTree as ET
import numpy as np

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, cdist
from azure.cosmos import CosmosClient, exceptions, PartitionKey

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
object_store_url = os.getenv("OBJECT_STORE")
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")


def download(filename, directory):
    download_url = f"{object_store_url}{directory}/{filename}"
    response = requests.get(download_url, auth=(username, password))
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print("File downloaded successfully")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        print(response.text)


def get_db_container():
    HOST = os.getenv("HOST")
    MASTER_KEY = os.getenv("MASTER_KEY")
    DATABASE_ID = os.getenv("DATABASE_ID")
    CONTAINER_ID = os.getenv("CONTAINER_ID")

    client = CosmosClient(HOST, MASTER_KEY)

    database = client.get_database_client(DATABASE_ID)

    container = database.get_container_client(CONTAINER_ID)

    return container


def get_song_from_db(user_query):
    container = get_db_container()
    query = "SELECT * FROM c"

    data = list(container.query_items(query=query, enable_cross_partition_query=True))

    embeddings_vector = []

    for val in data:
        embeddings_vector.append(val["vector"])

    user_vector = embedding_model.encode(user_query)
    user_vector = np.array(user_vector).reshape(1, -1)

    similarity = 1 - cdist(user_vector, np.array(embeddings_vector), "cosine")

    most_relevant_song_id = np.argsort(-similarity)[0][0]

    song = data[most_relevant_song_id]

    return song


def is_file_existing(filename):
    url = f"{object_store_url}cached_songs/{filename}"

    response = requests.request(
        "PROPFIND", url, auth=requests.auth.HTTPBasicAuth(username, password)
    )

    if response.status_code == 200:
        print("Cache hit")
        return True
    elif response.status_code == 207:
        namespace = {"dav": "DAV:"}
        root = ET.fromstring(response.content)

        for resp in root.findall("dav:response", namespace):
            href = resp.find("dav:href", namespace).text
            status = resp.find("dav:propstat/dav:status", namespace).text

            if filename in href:
                if "200 OK" in status:
                    print("Cache hit")
                    return True
                else:
                    return False
        return False
    else:
        print("Cache miss")
        return False


download("SiriVT.pth", "Siri")
download("added_IVF617_Flat_nprobe_1_SiriVT_v1.index", "Siri")


def cache_file(filename):
    upload_url = f"{object_store_url}cached_songs/{filename}"

    with open(f"./audio-outputs/{filename}", "rb") as file:
        response = requests.put(
            upload_url, data=file, auth=requests.auth.HTTPBasicAuth(username, password)
        )

    if response.status_code == 201:
        print("File cached successfully")
    elif response.status_code == 204:
        print("Cached File overwritten")
    else:
        print("Failed to upload file.")


def respond(message):
    song = get_song_from_db(message)
    song_file = song["audioPath"]
    octave = song["octave"]
    cached_song_file = f"{song_file[:-4]}.wav"

    print(cached_song_file)

    if is_file_existing(cached_song_file):
        download(cached_song_file, "cached_songs")
        return cached_song_file
    else:
        download(song_file, "songs")
        music_client = Client("r3gm/rvc_zero")
        result_list = music_client.predict(
            audio_files=[file(f"./{song_file}")],
            file_m=file("./SiriVT.pth"),
            pitch_alg="rmvpe+",
            pitch_lvl=octave,
            file_index=file("./added_IVF617_Flat_nprobe_1_SiriVT_v1.index"),
            index_inf=0.75,
            r_m_f=3,
            e_r=0.25,
            c_b_p=0.5,
            active_noise_reduce=False,
            audio_effects=False,
            api_name="/run",
        )

        result = result_list[0]
        source_path = result

        destination_path = os.path.join(os.getcwd(), cached_song_file)

        shutil.copy(source_path, destination_path)
        cache_file(cached_song_file)
        return destination_path


with gr.Blocks() as app:
    user_query = gr.Text("")

    with gr.Row():
        output = gr.Audio(label="Output Audio")

    btn = gr.Button(value="Submit")

    btn.click(respond, inputs=[user_query], outputs=[output])
    app.launch()
