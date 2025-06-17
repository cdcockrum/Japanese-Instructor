import gradio as gr
from huggingface_hub import InferenceClient

# Optional: use a public model to avoid 401
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# JLPT-level prompt generation
def level_to_prompt_japanese(level):
    return {
        "N5": "You are a kind Japanese tutor. Respond only to what the user asks. Use simple greetings, basic vocabulary, and Romaji. Provide English explanations and avoid off-topic content or voice suggestions.",
        "N4": "You are a patient Japanese tutor. Use short Japanese phrases with Romaji and English explanations. Introduce basic Kanji but avoid tangents or voice interaction.",
        "N3": "You are a helpful Japanese tutor. Use mostly Japanese with Romaji in parentheses. Include essential grammar or Kanji only if relevant to the user’s question.",
        "N2": "You are a fluent Japanese tutor. Respond only in Japanese unless clarification is requested. Use Kanji with Furigana or Hiragana, and remain on-topic.",
        "N1": "You are a Japanese language professor. Respond entirely in advanced Japanese using native expressions, Keigo (敬語), and idioms. Do not include translations or voice references unless explicitly asked."
    }.get(level, "You are a helpful Japanese tutor.")

# Chat handler
def respond(message, history, user_level, max_tokens, temperature, top_p):
    system_message = level_to_prompt_japanese(user_level)
    messages = [{"role": "system", "content": system_message}]
    
    # Format history
    if history and isinstance(history[0], tuple):
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    else:
        messages.extend(history)

    messages.append({"role": "user", "content": message})
    
    response = ""
    try:
        for msg in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = msg.choices[0].delta.content
            if token:
                response += token
            yield response
    except Exception as e:
        yield f"申し訳ありません！エラーが発生しました: {str(e)}"

# Gradio interface
with gr.Blocks(css="""
body {
  background-image: url('https://cdn-uploads.huggingface.co/production/uploads/67351c643fe51cb1aa28f2e5/GdA9eNQKjOQjE6q47km3l.jpeg');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}
.gradio-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 100vh;
  padding-top: 2rem;
  padding-bottom: 2rem;
}
#title-container {
  background-color: rgba(255, 255, 255, 0.85);  /* match chat panel */
  border-radius: 16px;
  padding: 1.5rem 2rem;
  margin: 2rem 0;
  width: fit-content;
  max-width: 350px;
  text-align: left;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  margin-left: 0rem;  /* Aligns left edge */
}

#title-container h1 {
  color: #222 !important;
  font-size: 4rem;
  font-family: 'Noto Sans JP', sans-serif;
  margin: 0;
}

#title-container .subtitle {
  font-size: 1.1rem;
  font-family: 'Noto Sans', sans-serif;
  color: #222 !important;
  margin-top: 0.5rem;
  margin-bottom: 0;
  width: 100%;
  display: block;
}
#chat-panel {
  background-color: rgba(255, 255, 255, 0.85);
  padding: 2rem;
  border-radius: 12px;
  justify-content: center;
  width: 100%;
  max-width: 700px;
  height: 70vh;
  box-shadow: 0 0 12px rgba(0, 0, 0, 0.3);
  overflow-y: auto;
}


.gradio-container .chatbot h1 {
   color: var(--custom-title-color) !important;
   font-family: 'Noto Sans', serif !important;
   font-size: 5rem !important;
   font-weight: bold !important;
   text-align: center !important;
   margin-bottom: 1.5rem !important;
   width: 100%;
}
""") as demo:
    
    #gr.Markdown("日本語の先生  Japanese Instructor", elem_id="custom-title")
    gr.HTML("""
    <div id="title-container">
      <h1>先生</h1>
      <p class="subtitle">Japanese Tutor</p>
    </div>
    """)
    
    with gr.Column(elem_id="chat-panel"):
        with gr.Accordion("Advanced Settings", open=False):
            user_level = gr.Dropdown(
                choices=["N5", "N4", "N3", "N2", "N1"],
                value="N5",
                label="Your Japanese Level (JLPT)"
            )
            max_tokens = gr.Slider(1, 2048, value=400, step=1, label="Response Length")
            temperature = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="Creativity")
            top_p = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Dynamic Text Sampling")

        gr.ChatInterface(
            respond,
            additional_inputs=[user_level, max_tokens, temperature, top_p],
            type="messages"
        )

if __name__ == "__main__":
    demo.launch()
