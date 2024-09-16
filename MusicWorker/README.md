---
title: MusicRetriever
emoji: 🐨
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

The Music worker retrieves the song metadata of the song whose description is most similar to user query. The metadata of the songs are stored in Cosmos db which also has the vector embedding of each song description on which cosine similarity search is performed.

The song is fetched from the object store and converted from the voice of the singer to voice of Siri. This server calls RVC Zero space (https://huggingface.co/spaces/r3gm/rvc_zero) which has a transformer that converts the song to the voice of Siri Model(https://huggingface.co/juuxn/RVCModels/resolve/main/Siri_VT_-_RVC_V1_-_100_Epoch.zip)

Additionally, a caching mechanism is implemented here for low latency.
