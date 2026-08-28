# Add at the VERY TOP of the script
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"CUDA_LAUNCH_BLOCKING set to: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")  # Verify

import torch
import time
import gradio as gr
from transformers import AutoModel, AutoTokenizer
import copy
import traceback
import threading 


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _sanitize_probabilities(probs: torch.Tensor) -> torch.Tensor:
    """Ensure probs (1D or 2D) contain finite, normalized values."""
    if not torch.is_floating_point(probs):
        return probs
    sanitized = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sanitized = sanitized.clamp_min(0.0)
    if sanitized.ndim == 1:
        total = sanitized.sum()
        if total <= 0:
            return torch.full_like(sanitized, 1.0 / sanitized.numel())
        return sanitized / total
    total = sanitized.sum(dim=-1, keepdim=True)
    zero_mask = total <= 0
    if zero_mask.any():
        sanitized = sanitized.clone()
        expanded_mask = zero_mask.expand(-1, sanitized.size(-1))
        sanitized[expanded_mask] = 1.0
        total = sanitized.sum(dim=-1, keepdim=True)
    return sanitized / total


def _flatten_probs_if_needed(probs: torch.Tensor):
    if probs.ndim <= 2:
        return probs, probs.shape
    orig_shape = probs.shape
    flat = probs.contiguous().view(-1, orig_shape[-1])
    return flat, orig_shape


def _reshape_samples_if_needed(samples: torch.Tensor, orig_shape, num_samples: int):
    if len(orig_shape) <= 2:
        return samples
    return samples.view(*orig_shape[:-1], num_samples)


_original_multinomial = torch.multinomial


def _safe_multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
    flat_probs, orig_shape = _flatten_probs_if_needed(probs)
    normalized = _sanitize_probabilities(flat_probs)
    samples = _original_multinomial(normalized, num_samples, replacement, generator=generator, out=out)
    return _reshape_samples_if_needed(samples, orig_shape, num_samples)


if not getattr(torch.multinomial, "_dream_safe", False):
    _safe_multinomial._dream_safe = True  # type: ignore[attr-defined]
    torch.multinomial = _safe_multinomial  # type: ignore[assignment]

# --- Model Loading ---
model_path = "Dream-org/Dream-v0-Instruct-7B"
device = select_device()
dtype_by_device = {
    "cuda": torch.bfloat16,
    "mps": torch.float16,
    "cpu": torch.float32,
}
dtype = dtype_by_device[device]
print(f"Using device: {device} (dtype={dtype})")

try:
    print(f"Loading model with {dtype}...")
    try:
        # transformers >= 4.56: `dtype` is the replacement for `torch_dtype`
        model = AutoModel.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
    except TypeError:
        # transformers < 4.56: `dtype` is not accepted yet
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    print(f"Model loaded successfully with {dtype}.")
except Exception as e:
    print(f"Fatal Error loading model: {e}")
    print(traceback.format_exc())
    exit()

# --- Tokenizer Loading ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Fatal Error loading tokenizer: {e}")
    print(traceback.format_exc())
    exit()

mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else -100
mask_token_str = "[MASK]"

# --- Move model to device ---
try:
    model = model.to(device=device, dtype=dtype).eval()
    print(f"Model moved to {device} and set to eval mode.")
except Exception as e:
    print(f"Fatal Error moving model to device: {e}")
    print(traceback.format_exc())
    exit()

# --- Helper Functions ---
def format_gradio_history_to_messages(history):
    messages = []
    if not history:
        return messages
    for pair in history:
        user_msg, assistant_msg = pair
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None:
            messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages

def add_user_message_to_gradio_history(history, message):
    if not history:
        history = []
    return history + [[message, None]]


def highlight_message(message, color="#CC6666"):
    return [(message, color)]

# --- Modified Main Generation Function with Real-Time Visualization ---
def dream_generate_with_visualization(history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp):
    print("\n--- Starting dream_generate_with_visualization ---")
    print(f"Parameters: max_new_tokens={max_new_tokens}, steps={steps}, temperature={temperature}, top_p={top_p}, top_k={top_k}, delay={delay}, alg={alg}, alg_temp={alg_temp}")

    messages_for_model = format_gradio_history_to_messages(history)

    try:
        inputs = tokenizer.apply_chat_template(
            messages_for_model, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        prompt_length = input_ids.shape[1]
        print(f"Prompt length: {prompt_length}, input_ids device: {input_ids.device}")
    except Exception as e:
        print(f"Error during input tokenization/processing: {e}")
        error_message = f"Input processing error: {e}"
        current_history = copy.deepcopy(history)
        if current_history:
            current_history[-1][1] = f"Error: {error_message}"
        else:
            current_history = [["System", f"Error: {error_message}"]]
        yield format_gradio_history_to_messages(current_history), highlight_message(error_message), current_history
        return

    # save histories
    visualization_token_states = []
    # Hook function for saving histories
    def my_generation_tokens_hook(step, x, logits):
        visualization_token_states.append(x[0].clone().cpu())
        return x

    effective_top_k = top_k if top_k > 0 else None

    # save final output or error
    output_container = {}

    def generation_func():
        try:
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                top_k=effective_top_k,
                alg=alg,
                alg_temp=alg_temp,
                generation_tokens_hook_func=my_generation_tokens_hook
            )
            output_container["output"] = output
        except Exception as e:
            output_container["error"] = e

    gen_thread = threading.Thread(target=generation_func)
    gen_thread.start()

    intermediate_history = copy.deepcopy(history)

    while len(visualization_token_states) == 0:
        time.sleep(0.01)  # wait for the first generation
    first_state = visualization_token_states[0]
    gen_length = first_state.shape[0] - prompt_length
    previous_tokens = [mask_token_id] * gen_length
    last_yielded = 0

    # check if new generations are added
    while gen_thread.is_alive() or last_yielded < len(visualization_token_states):
        current_length = len(visualization_token_states)
        while last_yielded < current_length:
            state_tensor = visualization_token_states[last_yielded]
            current_state_tensor = state_tensor[prompt_length:]
            current_tokens = current_state_tensor.tolist()
            colored_tokens = []
            # construct colorful token list
            for idx, token_id in enumerate(current_tokens):
                if token_id == mask_token_id:
                    colored_tokens.append((mask_token_str, "#444444"))
                else:
                    if previous_tokens[idx] == mask_token_id:
                        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                        colored_tokens.append((token_str, "#66CC66"))
                    else:
                        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                        colored_tokens.append((token_str, "#6699CC"))
            previous_tokens = current_tokens
            # update current states
            intermediate_history[-1][1] = f"⏳ Step {last_yielded}/{current_length - 1}"
            messages_for_chatbot_update = format_gradio_history_to_messages(intermediate_history)
            yield messages_for_chatbot_update, colored_tokens, history
            last_yielded += 1
        time.sleep(delay)

    # to ensure thread ends
    gen_thread.join()

    # check if there is any error
    if "error" in output_container:
        error_message = f"Error during model generation: {output_container['error']}"
        current_history = copy.deepcopy(history)
        if current_history:
            current_history[-1][1] = f"Error: {error_message}"
        else:
            current_history = [["System", f"Error: {error_message}"]]
        yield format_gradio_history_to_messages(current_history), highlight_message(error_message), current_history
        return

    # --- final result processing ---
    print("Processing final result...")
    try:
        output = output_container["output"]
        final_tokens_tensor = output.sequences[0][prompt_length:]
        final_tokens_list = final_tokens_tensor.tolist()
        colored_final = []
        for token_id in final_tokens_list:
            if token_id == mask_token_id:
                colored_final.append((mask_token_str, "#444444"))
            else:
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                colored_final.append((token_str, "#6699CC"))
        final_text = tokenizer.decode(final_tokens_list, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        history[-1][1] = final_text
        final_messages_for_chatbot = format_gradio_history_to_messages(history)
        print("Yielding final result.")
        yield final_messages_for_chatbot, colored_final, history
    except Exception as e:
        print(f"Error processing final output: {e}")
        error_message = f"Error processing final output: {e}"
        current_history = copy.deepcopy(history)
        if current_history:
            current_history[-1][1] = f"Error processing output: {error_message}"
        else:
            current_history = [["System", f"Error processing output: {error_message}"]]
        yield format_gradio_history_to_messages(current_history), highlight_message(error_message), current_history

    print("--- Exiting dream_generate_with_visualization normally ---")

# --- Bot Response Generator Wrapper ---
def bot_response_generator(history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp):
    if not history or history[-1][1] is not None:
        print("Skipping bot response: No history or last message already has a response.")
        yield format_gradio_history_to_messages(history), [], history
        return
    yield from dream_generate_with_visualization(history, max_new_tokens, steps, temperature, top_p, top_k, delay, alg, alg_temp)

# --- User Message Submission Handler ---
def user_message_submitted(message, history):
    if not message or not message.strip():
        return history, format_gradio_history_to_messages(history), ""
    new_history = add_user_message_to_gradio_history(history, message)
    messages_for_chatbot = format_gradio_history_to_messages(new_history)
    return new_history, messages_for_chatbot, ""

# --- Gradio UI ---
css = """
/* Make chatbot text selectable */
.gradio-container .prose ::selection { background-color: #ACE6FF; }
.gradio-container .prose ::-moz-selection { background-color: #ACE6FF; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dream Diffusion Model Demo (Text-to-Text)")
    gr.Markdown("Interact with the **Dream-v0-Instruct-7B** model in a multi-turn conversation and watch the diffusion process in real time.")
    gr.Markdown("Model link: [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)")

    chat_history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_display = gr.Chatbot(label="Chat", bubble_full_width=False, height=600, type="messages")
            with gr.Group():
                with gr.Row():
                    user_input_textbox = gr.Textbox(label="Your Message", placeholder="Type your message here...", scale=4, show_label=False, container=False)
                    send_button = gr.Button("Send", scale=1, variant="primary")
        with gr.Column(scale=2):
            vis_output_display = gr.HighlightedText(label="Diffusion Process Visualization", show_legend=True, combine_adjacent=False)

    with gr.Accordion("Generation Parameters", open=False):
        max_new_tokens_slider = gr.Slider(16, 512, value=128, step=16, label="Max New Tokens")
        steps_slider = gr.Slider(8, 512, value=128, step=8, label="Diffusion Steps")
        temperature_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature (0 = deterministic)")
        top_p_slider = gr.Slider(0.0, 1.0, value=0.95, step=0.05, label="Top-p (0 = disabled)")
        top_k_slider = gr.Slider(0, 100, value=0, step=1, label="Top-k (0 = disabled)")
        delay_slider = gr.Slider(0.0, 0.5, value=0.02, step=0.01, label="Visualization Delay (seconds)")
        alg_dropdown = gr.Dropdown(choices=["origin", "maskgit_plus", "topk_margin", "entropy"], value="entropy", label="Algorithm (alg)")
        alg_temp_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Algorithm Temperature (alg_temp)")

    clear_button = gr.Button("Clear Chat")

    def clear_conversation():
        return [], [], "", []

    clear_button.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chat_history_state, chatbot_display, user_input_textbox, vis_output_display],
        queue=False
    )

    generation_params = [max_new_tokens_slider, steps_slider, temperature_slider, top_p_slider, top_k_slider, delay_slider, alg_dropdown, alg_temp_slider]

    submit_event_args = dict(
        fn=user_message_submitted,
        inputs=[user_input_textbox, chat_history_state],
        outputs=[chat_history_state, chatbot_display, user_input_textbox],
    )

    bot_response_event_args = dict(
        fn=bot_response_generator,
        inputs=[chat_history_state] + generation_params,
        outputs=[chatbot_display, vis_output_display, chat_history_state]
    )

    submit_action = user_input_textbox.submit(**submit_event_args)
    submit_action.then(lambda: [], inputs=None, outputs=[vis_output_display])
    submit_action.then(**bot_response_event_args)

    send_action = send_button.click(**submit_event_args)
    send_action.then(lambda: [], inputs=None, outputs=[vis_output_display])
    send_action.then(**bot_response_event_args)

if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=1).launch(share=True, debug=True)