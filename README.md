# emoji-suggest

A simple API which recommends a single emoji given a string of text, using embeddings and [CLIP](https://github.com/openai/CLIP).

Simply send a GET request to the root URL with an appended string (`/hello%20world`) and you'll get a JSON response with an emoji.